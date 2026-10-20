"""Request lifecycle: bypass -> normalize -> retrieve -> verify -> serve -> store -> log."""
import time
from dataclasses import dataclass, field

import numpy as np

from . import patcher
from .context import context_key, has_unresolved_reference, split_messages
from .cost import cost_usd
from .index import top_k
from .normalize import normalize
from .store import Store
from .verifier.borderline import same_answer
from .verifier.cross import CosineScorer, Scorer
from .verifier.guard import Guard
from .verifier.policy import HIT, MISS, NEAR_HIT, Decision, decide, needs_guard
from .verifier.rules import extract_differences, is_time_sensitive

BYPASS = "bypass"


@dataclass
class Result:
    text: str
    tier: str
    reason: str
    source: str | None = None
    fallback: bool = False
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    extra: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self, cfg: dict, store: Store, embedder, large, small=None, scorer: Scorer | None = None,
                 guard: Guard | None = None):
        self.cfg, self.store, self.embedder = cfg, store, embedder
        self.guard = guard
        self.large, self.small = large, small
        self.scorer = scorer or CosineScorer()

    def handle(self, messages: list[dict], scope: str = "default", bypass: bool = False,
               temperature: float | None = None, has_tools: bool = False, **params) -> Result:
        t0 = time.perf_counter()
        query, prior = split_messages(messages)
        cfg = self.cfg
        pricing = cfg.get("pricing", {})

        reason = None
        if bypass:
            reason = "bypass: header"
        elif has_tools and cfg["bypass"]["with_tools"]:
            reason = "bypass: tools"
        elif temperature is not None and temperature > cfg["bypass"]["max_temperature"]:
            reason = "bypass: high temperature"
        if reason:
            c = self.large.complete(messages, **({"temperature": temperature} if temperature is not None else {}), **params)
            return self._finish(scope, query, BYPASS, reason, c.text, None, None, None, [], False,
                                cost_usd(pricing, "large", c.in_tokens, c.out_tokens), t0)

        ctx = context_key(prior, cfg["multi_turn"]["context_turns"])
        time_sensitive = is_time_sensitive(query)
        unresolved = bool(prior) is False and has_unresolved_reference(query) and len(query.split()) < 8

        decision, cand, cos = Decision(MISS, "miss: no candidates"), None, None
        judge_cost = 0.0
        qvec = None
        if time_sensitive:
            decision = Decision(MISS, "miss: time-sensitive")
        elif unresolved:
            decision = Decision(MISS, "miss: unresolved reference")
        else:
            qnorm = normalize(query)
            qvec = self.embedder.encode(qnorm)
            entries = self.store.candidates(scope, ctx)
            vecs = [np.frombuffer(e.embedding, dtype=np.float32) for e in entries if e.embedding]
            entries = [e for e in entries if e.embedding]
            best = None
            for pos, c in top_k(qvec, vecs, cfg["verifier"]["top_k"]):
                e = entries[pos]
                diffs = extract_differences(e.query, query, cfg["verifier"].get("strict_content_check", True))
                d = decide(diffs, self.scorer.score(e.query, query, c),
                           cfg["verifier"]["hit_threshold"], cfg["verifier"]["near_threshold"], cosine=c)
                if self._borderline(d, c):
                    contra = self.guard.contradiction(e.query, query) if self.guard else 0.0
                    if contra >= cfg["verifier"].get("contradiction_threshold", 0.5):
                        d = Decision(MISS, f"miss: contradiction ({contra:.2f})", d.score, d.differences)  # free check first
                    else:
                        ok, comp = same_answer(self.small, e.query, query)
                        judge_cost += cost_usd(pricing, "small", comp.in_tokens, comp.out_tokens)
                        if ok:
                            d = Decision(HIT, "hit: confirmed by judge", d.score, [])
                if self.guard and needs_guard(d):
                    contra = self.guard.contradiction(e.query, query)
                    if contra >= cfg["verifier"].get("contradiction_threshold", 0.5):
                        d = Decision(MISS, f"miss: contradiction ({contra:.2f})", d.score, d.differences)
                rank = {HIT: 2, NEAR_HIT: 1, MISS: 0}[d.tier]
                if best is None or (rank, d.score) > (best[0], best[1].score):
                    best = (rank, d, e, c)
            if best:
                _, decision, cand, cos = best

        cost, fell_back = judge_cost, False
        if decision.tier == HIT and cand:
            self.store.touch(cand.id)
            return self._finish(scope, query, HIT, decision.reason, cand.answer, cand.id, cos, decision.score,
                                decision.differences, False, judge_cost, t0)

        if decision.tier == NEAR_HIT and cand and self.small and cfg["patcher"]["enabled"]:
            patched, c = patcher.patch(self.small, cand.query, cand.answer, query, decision.differences,
                                       cfg["patcher"]["max_length_ratio"])
            cost += cost_usd(pricing, "small", c.in_tokens, c.out_tokens)
            if patched is not None:
                self.store.touch(cand.id)
                if cfg["patcher"]["cache_patched_answers"]:
                    self._store(scope, ctx, query, patched, self.small.model, qvec)
                return self._finish(scope, query, NEAR_HIT, decision.reason, patched, cand.id, cos,
                                    decision.score, decision.differences, False, cost, t0)
            fell_back = True
            decision = Decision(MISS, "miss: near_hit fallback: CANNOT_PATCH", decision.score, decision.differences)
        elif decision.tier == NEAR_HIT:
            decision = Decision(MISS, "miss: near_hit but patcher disabled", decision.score, decision.differences)

        c = self.large.complete(messages, **({"temperature": temperature} if temperature is not None else {}), **params)
        cost += cost_usd(pricing, "large", c.in_tokens, c.out_tokens)
        cacheable = not time_sensitive and not unresolved and bool(c.text)
        if cacheable:
            volatile, vcost = self._is_volatile(query)
            cost += vcost
            cacheable = not volatile
        if cacheable:
            if qvec is None:
                qvec = self.embedder.encode(normalize(query))
            self._store(scope, ctx, query, c.text, self.large.model, qvec)
        return self._finish(scope, query, MISS, decision.reason, c.text, cand.id if fell_back and cand else None,
                            cos, decision.score, decision.differences, fell_back, cost, t0)

    def _borderline(self, d: Decision, cosine: float) -> bool:
        """A pair the scorer is unsure about and no rule objects to: worth one small-model opinion."""
        v = self.cfg["verifier"]
        return bool(v.get("borderline_judge") and self.small and d.tier == MISS and not d.differences
                    and cosine >= v["near_threshold"] and v.get("borderline_low", 0.3) <= d.score < v["hit_threshold"])

    def _is_volatile(self, query: str) -> tuple[bool, float]:
        """Ask the small model whether the correct answer changes over time. Runs once per stored answer."""
        if not (self.cfg["cache"].get("volatility_check") and self.small):
            return False, 0.0
        c = self.small.complete([{"role": "user", "content":
            "Does the correct answer to this question change over time (news, prices, scores, weather, "
            "current office-holders, availability, anything 'right now')? Reply with only YES or NO.\n\n"
            f"Question: {query}"}])
        cost = cost_usd(self.cfg.get("pricing", {}), "small", c.in_tokens, c.out_tokens)
        return c.text.strip().upper().startswith("YES"), cost

    def _store(self, scope, ctx, query, answer, model, qvec):
        self.store.add_entry(scope, ctx, query, normalize(query), answer, model,
                             embedding=qvec.astype(np.float32).tobytes(),
                             ttl=self.cfg["cache"].get("default_ttl_seconds"))

    def _finish(self, scope, query, tier, reason, text, source, cos, score, diffs, fell_back, cost, t0) -> Result:
        ms = (time.perf_counter() - t0) * 1000
        self.store.log_decision(scope, query, tier, reason, source, cos, score, diffs, fell_back, ms, cost)
        return Result(text, tier, reason, source, fell_back, cost, ms)
