"""Re-judge the pilot's served HITS and print the judge's raw replies, to check the judge itself.

  python -m eval.judge_debug --config configs/anthropic.yaml --cap 0.12

A hit returns the seed's cached answer unchanged, so a good paraphrase should be judged acceptable.
Costs ~$0.06: for each hit it regenerates the seed answer and a fresh reference (large model).
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gistcache.config import load_config
from gistcache.providers import build_provider

from .judge import judge
from .spend import Tracker

HERE = Path(__file__).parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/anthropic.yaml")
    p.add_argument("--cap", type=float, default=0.12)
    p.add_argument("--max-tokens", type=int, default=300)
    a = p.parse_args()
    cfg = load_config(a.config)
    t = Tracker(a.cap, cfg["pricing"])
    large = t.wrap(build_provider(cfg["providers"]["large"]), "large")
    small = t.wrap(build_provider(cfg["providers"]["small"]), "small")
    rows = [json.loads(l) for l in (HERE / "results" / "pilot.jsonl").read_text().splitlines()]
    hits = [r for r in rows if r["tier"] == "hit"]

    def one(r):
        ask = lambda q: large.complete([{"role": "user", "content": q}], max_tokens=a.max_tokens).text
        cached, ref = ask(r["seed"]), ask(r["query"])
        ok, why = judge(small, r["query"], ref, cached)
        return r, cached, ref, ok, why

    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(one, hits))
    for r, cached, ref, ok, why in results:
        print(f"\n=== {r['query']!r}\n    seed: {r['seed']!r}\n    verdict={ok}  raw judge reply: {why!r}")
        print(f"    cached answer starts: {cached[:130]!r}\n    reference starts:     {ref[:130]!r}")
    print(f"\nacceptable: {sum(1 for x in results if x[3])}/{len(results)}   spent ${t.total:.4f}")


if __name__ == "__main__":
    main()
