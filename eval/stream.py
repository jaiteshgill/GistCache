"""End-to-end evaluation: seeds -> labelled variants -> GistCache and threshold caches on the same stream ->
judge -> report.

  python -m eval.stream --dry-run                                                      # fake providers, free
  python -m eval.stream --config configs/anthropic.yaml --n 24 --cap 0.35              # hand-written seeds
  python -m eval.stream --config configs/anthropic.yaml --seeds-file eval/data/seeds_lmsys.json --n 120 --cap 2

Spend is hard-capped. Phases: variants (small model), GistCache stream (stops at 55% of the cap), threshold
baselines on the same stream (stop at 75%), judging (the rest). Each distinct large-model request is paid for
once (see eval.baseline.Memo). With --seeds-file the published report contains no prompt text, because real
datasets such as LMSYS-Chat-1M may not be redistributed; a .local.md copy with text is gitignored.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import random
import threading
from collections import Counter, defaultdict
from pathlib import Path

from gistcache.config import load_config
from gistcache.pipeline import Pipeline
from gistcache.providers import Completion, build_provider
from gistcache.store import Store

from .baseline import Memo, ThresholdCache
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
    large, small = Memo(tracker.wrap(large, "large")), tracker.wrap(small, "small")
    return Pipeline(cfg, Store(":memory:"), emb, large, small, scorer, guard), large, small, emb


def build_queries(variants: dict, rng: random.Random):
    seeds = list(variants)
    rng.shuffle(seeds)
    first = [{"seed": s, "kind": "seed", "query": s} for s in seeds]
    rest = [{"seed": s, "kind": k, "query": v[k]} for s, v in variants.items() for k in ("paraphrase", "patchable", "critical")]
    rng.shuffle(rest)
    return first + rest  # every seed is cached before any of its variants is asked


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))] if xs else 0.0


def system_row(name, rows, n_total, served_tiers):
    served = [r for r in rows if r["tier"] in served_tiers]
    judged = [r for r in served if r.get("ok") is not None]
    bad = sum(1 for r in judged if not r["ok"])
    ms = [r["ms"] for r in rows]
    cost = sum(r["cost"] for r in rows)
    return (f"| {name} | {100 * len(served) / max(n_total, 1):.0f}% | {sum(r['ok'] for r in judged)}/{len(judged)} | "
            f"{bad} ({100 * bad / max(n_total, 1):.1f} per 100) | ${1000 * cost / max(n_total, 1):.2f} | "
            f"{pct(ms, 50) / 1000:.2f} s | {pct(ms, 95) / 1000:.2f} s |")


def report(rows, base_rows, tracker, mean_large, large_ms, stopped_early, n_unjudged, title, redact) -> str:
    L = []
    n = len(rows)
    L.append(f"# {title}\n\n{n} queries run{' (stopped early: budget)' if stopped_early else ''}. "
             f"API spend ${tracker.total:.4f} of ${tracker.cap_usd:g} cap "
             f"(large {tracker.calls['large']} calls ${tracker.spent['large']:.4f}, "
             f"small {tracker.calls['small']} calls ${tracker.spent['small']:.4f}).\n")
    L.append("## End to end (same query stream for every system)\n")
    L.append("| System | Large model avoided | Reused answers judged acceptable | Wrongly reused | Cost / 1k requests | p50 latency | p95 latency |")
    L.append("|---|---|---|---|---|---|---|")
    L.append(f"| No cache | 0% | n/a | 0 | ${1000 * mean_large:.2f} | {pct(large_ms, 50) / 1000:.2f} s | {pct(large_ms, 95) / 1000:.2f} s |")
    for t, brs in base_rows.items():
        L.append(system_row(f"Threshold cache (cosine >= {t:g})", brs, len(brs), ("hit",)))
    L.append(system_row("**GistCache**", rows, n, ("hit", "near_hit")))
    L.append("\nNo-cache cost and latency are the mean and percentiles of real large-model calls in this run. "
             "\"Wrongly reused\" counts reused answers the judge rejected.\n")

    L.append("## GistCache tier by query type\n")
    L.append("| Query type | n | Expected | hit | near_hit | miss | Over-reuse |\n|---|---|---|---|---|---|---|")
    for kind in ("seed", "paraphrase", "patchable", "critical"):
        rs = [r for r in rows if r["kind"] == kind]
        if not rs:
            continue
        c = Counter(r["tier"] for r in rs)
        over = sum(RANK[r["tier"]] > RANK[EXPECT[kind]] for r in rs)
        L.append(f"| {kind} | {len(rs)} | {EXPECT[kind]} | {c['hit']} | {c['near_hit']} | {c['miss'] + c['bypass']} | {over} |")
    if base_rows:
        L.append("\n## Threshold cache hits by query type\n")
        L.append("| Threshold | paraphrase hits | patchable hits | critical hits (should be 0) |\n|---|---|---|---|")
        for t, brs in base_rows.items():
            h = Counter(r["kind"] for r in brs if r["tier"] == "hit")
            k = Counter(r["kind"] for r in brs)
            L.append(f"| {t:g} | {h['paraphrase']}/{k['paraphrase']} | {h['patchable']}/{k['patchable']} | {h['critical']}/{k['critical']} |")

    served = [r for r in rows if r["tier"] in ("hit", "near_hit")]
    judged = [r for r in served if r.get("ok") is not None]
    unclear = [r for r in served if "why" in r and r.get("ok") is None]  # empty or no YES/NO
    L.append("\n## GistCache: quality of reused answers (LLM judge vs a large-model reference)\n")
    for tier in ("hit", "near_hit"):
        js = [r for r in judged if r["tier"] == tier]
        L.append(f"- {tier}: {sum(r['ok'] for r in js)}/{len(js)} acceptable"
                 f" ({sum(1 for r in served if r['tier'] == tier) - len(js)} not judged)")
    for r in (r for r in judged if not r["ok"]):
        what = f"[{r['tier']}, {r['kind']}] ({r['reason']})" if redact else \
            f"[{r['tier']}, {r['kind']}]: {r['query']!r} ({r['reason']}) -> {r.get('why', '')[:140]!r}"
        L.append(f"  - NOT acceptable {what}")
    if unclear:
        L.append(f"  - {len(unclear)} judge replies had no YES/NO and were excluded")
    pipe_cost = sum(r["cost"] for r in rows)
    base = n * mean_large
    L.append("\n## GistCache cost and latency\n")
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
    if redact:
        L.append("\nPrompt text is omitted because the source dataset may not be redistributed.")
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/anthropic.yaml")
    p.add_argument("--n", type=int, default=24, help="number of seeds")
    p.add_argument("--seeds-file", help="JSON from eval.seeds_build; default: the 40 hand-written seeds")
    p.add_argument("--baselines", default="0.80,0.90", help="threshold-cache cosine thresholds ('' for none)")
    p.add_argument("--cap", type=float, default=0.45, help="hard USD spend cap")
    p.add_argument("--max-tokens", type=int, default=300, help="cap on every answer length")
    p.add_argument("--rng", type=int, default=7)
    p.add_argument("--workers", type=int, default=8, help="parallel calls for variants and judging")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    cfg = load_config(a.config)
    cfg["cache"]["path"] = ":memory:"
    if a.dry_run:
        cfg["pricing"] = {"large": {"input": 2.0, "output": 10.0}, "small": {"input": 0.1, "output": 0.5}}
    tracker = Tracker(a.cap, cfg["pricing"])
    pipe, large, small, emb = build(cfg, a.dry_run, tracker)
    rng = random.Random(a.rng)
    if a.seeds_file:
        pool = json.loads(Path(a.seeds_file).read_text())["seeds"]
        tag = Path(a.seeds_file).stem.replace("seeds_", "")
        name, title, redact = f"eval_{tag}", f"GistCache evaluation ({tag} seeds)", True
    else:
        pool, tag = SEEDS, "handwritten"
        name, title, redact = "pilot", "GistCache pilot", False
    if a.dry_run:
        name, title = name + "_dry", title + " - DRY RUN (fake providers)"
    seeds = rng.sample(pool, min(a.n, len(pool)))
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)

    vname = "variants_dry.json" if a.dry_run else ("variants.json" if tag == "handwritten" else f"variants_{tag}.json")
    variants = generate(small, seeds, HERE / "data" / vname, a.workers)
    queries = build_queries(variants, rng)
    print(f"{len(variants)} seeds -> {len(queries)} queries; variants cost so far ${tracker.total:.4f}")

    rows, stopped = [], False
    for i, q in enumerate(queries):
        if tracker.total >= 0.55 * a.cap:
            stopped = True
            print(f"stopping GistCache stream at {i}/{len(queries)}: 55% of cap used")
            break
        try:
            r = pipe.handle([{"role": "user", "content": q["query"]}], scope="eval", max_tokens=a.max_tokens)
        except BudgetExceeded as e:
            stopped = True
            print("budget:", e)
            break
        rows.append({**q, "tier": r.tier, "reason": r.reason, "cost": r.cost_usd, "ms": r.latency_ms, "text": r.text})
        print(f"[{i + 1}/{len(queries)}] {q['kind']:10} -> {r.tier:8} {r.reason[:48]}")

    base_rows = {}
    for t in [float(x) for x in a.baselines.split(",") if x.strip()]:
        tc, brs = ThresholdCache(emb, large, t, cfg["pricing"]), []
        for q in queries[:len(rows)]:  # exactly the queries GistCache saw, in the same order
            if tracker.total >= 0.75 * a.cap:
                stopped = True
                print(f"stopping threshold {t:g} at {len(brs)}/{len(rows)}: 75% of cap used")
                break
            try:
                brs.append({**q, **tc.handle(q["query"], a.max_tokens)})
            except BudgetExceeded:
                stopped = True
                break
        base_rows[t] = brs
        h = Counter(r["kind"] for r in brs if r["tier"] == "hit")
        print(f"threshold cache {t:g}: {sum(h.values())} hits ({h['critical']} on critical variants)")

    verdicts, lock = {}, threading.Lock()

    def grade(r):
        key = (r["query"], r["text"])  # the same answer for the same query is judged once, whatever served it
        try:
            with lock:
                done = verdicts.get(key)
            if done is None:
                ref = large.complete([{"role": "user", "content": r["query"]}], max_tokens=a.max_tokens).text
                done = judge(small, r["query"], ref, r["text"])
                with lock:
                    verdicts[key] = done
            r["ok"], r["why"] = done
            return 0
        except BudgetExceeded:
            return 1

    to_judge = [r for r in rows if r["tier"] in ("hit", "near_hit")]
    to_judge += [r for brs in base_rows.values() for r in brs if r["tier"] == "hit"]
    print(f"judging {len(to_judge)} reused answers ({a.workers} in parallel)...")
    with ThreadPoolExecutor(a.workers) as ex:
        n_unjudged = sum(ex.map(grade, to_judge))

    mean_large = tracker.spent["large"] / tracker.calls["large"] if tracker.calls["large"] else 0.0
    md = report(rows, base_rows, tracker, mean_large, large.latencies(), stopped, n_unjudged, title, redact)
    (out_dir / f"{name}.md").write_text(md)
    if redact:
        (out_dir / f"{name}.local.md").write_text(report(rows, base_rows, tracker, mean_large, large.latencies(),
                                                         stopped, n_unjudged, title, False))
    (out_dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("\n" + md)
    if a.dry_run:
        # Rough real-cost projection: real calls this dry run would make, priced at the pilot's per-call averages
        # (large $0.00301, small $0.000116). Fake providers produce different tiers, so treat it as a ballpark.
        est = tracker.calls["large"] * 0.00301 + tracker.calls["small"] * 0.000116
        print(f"projected real cost for this run: about ${est:.2f} "
              f"({tracker.calls['large']} large calls, {tracker.calls['small']} small calls)")


if __name__ == "__main__":
    main()
