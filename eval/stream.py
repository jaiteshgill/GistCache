"""End-to-end pilot: seeds -> labelled variants -> GistCache pipeline -> judge -> report.

  python -m eval.stream --config configs/anthropic.yaml --n 24 --cap 0.45        # real, spends money
  python -m eval.stream --dry-run                                                 # fake providers, free

Spend is hard-capped. Order: variants (small), stream (stops at 70% of cap), judging (rest of cap).
"""
import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from gistcache.config import load_config
from gistcache.pipeline import Pipeline
from gistcache.providers import Completion, build_provider
from gistcache.store import Store

from .judge import judge
from .seeds import SEEDS
from .spend import BudgetExceeded, Tracker
from .variants import generate

HERE = Path(__file__).parent
RANK = {"miss": 0, "bypass": 0, "near_hit": 1, "hit": 2}
EXPECT = {"seed": "miss", "paraphrase": "hit", "patchable": "near_hit", "critical": "miss"}


class FakeProvider:
    """Offline stand-in: deterministic answers, variants and a lenient judge."""
    def __init__(self, model):
        self.model = model

    def complete(self, messages, **params):
        text = messages[-1]["content"]
        if "building a test set" in text:
            seed = text.split("Prompt:", 1)[1].strip()
            out = json.dumps({"paraphrase": seed.replace("How do I", "How can I"),
                              "patchable": seed.rstrip("?.") + ", briefly",
                              "critical": seed.rstrip("?.") + " in Rust"})
        elif "grading an answer" in text:
            out = "YES"
        elif "change over time" in text:
            out = "NO"
        elif "adapt a cached answer" in text.lower():
            out = "Adapted: " + text.split("Cached answer:", 1)[1].split("New query:")[0].strip()
        else:
            out = f"Answer to: {text}. " + "Detailed explanation. " * 8
        return Completion(out, len(text) // 4, len(out) // 4)


def build(cfg, dry: bool, tracker: Tracker):
    if dry:
        from gistcache.embed import HashEmbedder
        emb, scorer, guard = HashEmbedder(), None, None
        large, small = FakeProvider("large"), FakeProvider("small")
    else:
        from gistcache.embed import SentenceTransformerEmbedder
        from gistcache.verifier.cross import CrossEncoderScorer
        from gistcache.verifier.guard import NLIGuard
        e = cfg.get("embedder", {})
        emb = SentenceTransformerEmbedder(e.get("model", "sentence-transformers/all-MiniLM-L6-v2"), e.get("device", "cpu"))
        scorer = CrossEncoderScorer(cfg["verifier"].get("cross_encoder", "cross-encoder/quora-distilroberta-base"))
        guard = NLIGuard(cfg["verifier"]["nli_guard"]) if cfg["verifier"].get("nli_guard") else None
        large, small = build_provider(cfg["providers"]["large"]), build_provider(cfg["providers"]["small"])
    large, small = tracker.wrap(large, "large"), tracker.wrap(small, "small")
    return Pipeline(cfg, Store(":memory:"), emb, large, small, scorer, guard), large, small


def build_queries(variants: dict, rng: random.Random):
    seeds = list(variants)
    rng.shuffle(seeds)
    first = [{"seed": s, "kind": "seed", "query": s} for s in seeds]
    rest = [{"seed": s, "kind": k, "query": v[k]} for s, v in variants.items() for k in ("paraphrase", "patchable", "critical")]
    rng.shuffle(rest)
    return first + rest  # every seed is cached before any of its variants is asked


def report(rows, tracker, mean_large, stopped_early, n_unjudged) -> str:
    L = []
    n = len(rows)
    L.append(f"# GistCache pilot\n\n{n} queries run{' (stopped early: budget)' if stopped_early else ''}. "
             f"API spend ${tracker.total:.4f} of ${tracker.cap_usd:g} cap "
             f"(large {tracker.calls['large']} calls ${tracker.spent['large']:.4f}, "
             f"small {tracker.calls['small']} calls ${tracker.spent['small']:.4f}).\n")
    L.append("## Tier by query type\n")
    L.append("| Query type | n | Expected | hit | near_hit | miss | Over-reuse |\n|---|---|---|---|---|---|---|")
    for kind in ("seed", "paraphrase", "patchable", "critical"):
        rs = [r for r in rows if r["kind"] == kind]
        if not rs:
            continue
        c = Counter(r["tier"] for r in rs)
        over = sum(RANK[r["tier"]] > RANK[EXPECT[kind]] for r in rs)
        L.append(f"| {kind} | {len(rs)} | {EXPECT[kind]} | {c['hit']} | {c['near_hit']} | {c['miss'] + c['bypass']} | {over} |")
    served = [r for r in rows if r["tier"] in ("hit", "near_hit")]
    judged = [r for r in served if r.get("ok") is not None]
    L.append("\n## Quality of reused answers (LLM judge vs fresh reference)\n")
    for tier in ("hit", "near_hit"):
        js = [r for r in judged if r["tier"] == tier]
        L.append(f"- {tier}: {sum(r['ok'] for r in js)}/{len(js)} acceptable"
                 f" ({sum(1 for r in served if r['tier'] == tier) - len(js)} not judged)")
    bad = [r for r in judged if not r["ok"]]
    for r in bad:
        L.append(f"  - NOT acceptable [{r['tier']}, {r['kind']}]: {r['query']!r} (served from {r['reason']})")
    pipe_cost = sum(r["cost"] for r in rows)
    base = n * mean_large
    L.append("\n## Cost and latency\n")
    L.append(f"- GistCache pipeline cost: ${pipe_cost:.4f}")
    L.append(f"- No-cache estimate ({n} x mean large call ${mean_large:.5f}): ${base:.4f}")
    if base:
        L.append(f"- Saved: {100 * (1 - pipe_cost / base):.0f}%")
    by = defaultdict(list)
    for r in rows:
        by[r["tier"]].append(r["ms"])
    for t, ms in sorted(by.items()):
        L.append(f"- mean latency {t}: {sum(ms) / len(ms):.0f} ms (n={len(ms)})")
    if n_unjudged:
        L.append(f"\nNote: {n_unjudged} reused answers were not judged (budget).")
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/anthropic.yaml")
    p.add_argument("--n", type=int, default=24, help="number of seeds")
    p.add_argument("--cap", type=float, default=0.45, help="hard USD spend cap")
    p.add_argument("--max-tokens", type=int, default=300, help="cap on every answer length")
    p.add_argument("--rng", type=int, default=7)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    cfg = load_config(a.config)
    cfg["cache"]["path"] = ":memory:"
    if a.dry_run:
        cfg["pricing"] = {"large": {"input": 2.0, "output": 10.0}, "small": {"input": 0.1, "output": 0.5}}
    tracker = Tracker(a.cap, cfg["pricing"])
    pipe, large, small = build(cfg, a.dry_run, tracker)
    rng = random.Random(a.rng)
    seeds = rng.sample(SEEDS, min(a.n, len(SEEDS)))
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)

    vpath = HERE / "data" / ("variants_dry.json" if a.dry_run else "variants.json")
    variants = generate(small, seeds, vpath)
    queries = build_queries(variants, rng)
    print(f"{len(variants)} seeds -> {len(queries)} queries; variants cost so far ${tracker.total:.4f}")

    rows, stopped = [], False
    for i, q in enumerate(queries):
        if tracker.total >= 0.7 * a.cap:
            stopped = True
            print(f"stopping stream at {i}/{len(queries)}: 70% of cap used")
            break
        try:
            r = pipe.handle([{"role": "user", "content": q["query"]}], scope="pilot", max_tokens=a.max_tokens)
        except BudgetExceeded as e:
            stopped = True
            print("budget:", e)
            break
        rows.append({**q, "tier": r.tier, "reason": r.reason, "cost": r.cost_usd, "ms": r.latency_ms, "text": r.text})
        print(f"[{i + 1}/{len(queries)}] {q['kind']:10} -> {r.tier:8} {r.reason[:48]}")

    n_unjudged = 0
    for r in rows:
        if r["tier"] not in ("hit", "near_hit"):
            continue
        try:
            ref = large.complete([{"role": "user", "content": r["query"]}], max_tokens=a.max_tokens).text
            r["ok"] = judge(small, r["query"], ref, r["text"])
        except BudgetExceeded:
            n_unjudged += 1

    mean_large = tracker.spent["large"] / tracker.calls["large"] if tracker.calls["large"] else 0.0
    md = report(rows, tracker, mean_large, stopped, n_unjudged)
    name = "pilot_dry" if a.dry_run else "pilot"
    (out_dir / f"{name}.md").write_text(md)
    (out_dir / f"{name}.jsonl").write_text("".join(json.dumps({k: v for k, v in r.items() if k != "text"}) + "\n" for r in rows))
    print("\n" + md)


if __name__ == "__main__":
    main()
