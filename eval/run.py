"""Verifier eval on the hard set: GistCache policy vs a plain cosine-threshold cache.

  python -m eval.run --split dev            # tune on dev
  python -m eval.run --split test           # report once at the end
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from gistcache.config import load_config
from gistcache.normalize import normalize
from gistcache.verifier.policy import decide
from gistcache.verifier.rules import extract_differences, is_time_sensitive


RANK = {"miss": 0, "near_hit": 1, "hit": 2}  # reuse level; over-reuse = more permissive than expected


def load_models(real: bool):
    if real:
        from gistcache.embed import SentenceTransformerEmbedder
        from gistcache.verifier.cross import CrossEncoderScorer
        return SentenceTransformerEmbedder(), CrossEncoderScorer()
    from gistcache.embed import HashEmbedder
    from gistcache.verifier.cross import CosineScorer
    return HashEmbedder(), CosineScorer()


def evaluate(rows, embedder, scorer, hit_t, near_t, baseline_t):
    out = []
    for r in rows:
        a, b = r["cached"], r["new"]
        cos = float(embedder.encode(normalize(a)) @ embedder.encode(normalize(b)))
        score = scorer.score(a, b, cos)
        if is_time_sensitive(a) or is_time_sensitive(b):
            tier, reason = "miss", "miss: time-sensitive"
        else:
            d = decide(extract_differences(a, b), score, hit_t, near_t, cosine=cos)
            tier, reason = d.tier, d.reason
        base = "hit" if cos >= baseline_t else "miss"
        out.append({**r, "cos": cos, "score": score, "got": tier, "reason": reason, "baseline": base})
    return out


def report(res, show_errors):
    cats = defaultdict(list)
    for r in res:
        cats[r["category"]].append(r)
    print(f"{'category':16}{'n':>4}  {'threshold cache':>16}  {'GistCache':>10}  {'over-reuse (GC)':>18}")
    tot = defaultdict(int)
    for c, rs in cats.items():
        # Baseline can only serve or not serve; it is correct when it serves iff the expected tier is hit.
        b = sum((r["baseline"] == "hit") == (r["expected"] == "hit") for r in rs)
        g = sum(r["got"] == r["expected"] for r in rs)
        unsafe = sum(RANK[r["got"]] > RANK[r["expected"]] for r in rs)
        print(f"{c:16}{len(rs):>4}  {b:>10}/{len(rs):<4}  {g:>5}/{len(rs):<4}  {unsafe:>18}")
        tot["n"] += len(rs); tot["b"] += b; tot["g"] += g; tot["u"] += unsafe
    print(f"{'TOTAL':16}{tot['n']:>4}  {tot['b']:>10}/{tot['n']:<4}  {tot['g']:>5}/{tot['n']:<4}  {tot['u']:>18}")
    if show_errors:
        print("\nGistCache errors:")
        for r in res:
            if r["got"] != r["expected"]:
                print(f"  [{r['id']}] expected {r['expected']}, got {r['got']} ({r['reason']}; cos={r['cos']:.2f} "
                      f"score={r['score']:.2f})\n      {r['cached']!r} -> {r['new']!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.add_argument("--config")
    p.add_argument("--hash", action="store_true", help="use the dependency-free fallback models")
    p.add_argument("--baseline-threshold", type=float, default=0.85)
    p.add_argument("--errors", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config)
    rows = [json.loads(l) for l in (Path(__file__).parent / "hard_set.jsonl").read_text().splitlines()]
    if a.split != "all":
        rows = [r for r in rows if r["split"] == a.split]
    emb, sc = load_models(not a.hash)
    v = cfg["verifier"]
    print(f"split={a.split} pairs={len(rows)} hit_t={v['hit_threshold']} near_t={v['near_threshold']} "
          f"baseline_t={a.baseline_threshold}\n")
    report(evaluate(rows, emb, sc, v["hit_threshold"], v["near_threshold"], a.baseline_threshold), a.errors)


if __name__ == "__main__":
    main()
