"""A/B check of the patcher prompt: does the new prompt refuse less on pure reformatting without
inventing facts?

  python -m eval.patcher_check                      # free: show the cases
  python -m eval.patcher_check --live --cap 0.08    # needs an API key; ~$0.04

Real cases come from the last pilot (eval/results/pilot.jsonl): every patchable variant that was refused plus
some that succeeded. Their cached answers are regenerated once with the large model (cached on disk, ~$0.04).
Synthetic controls use hand-written cached answers, so they cost almost nothing:
  MUST_PATCH  - pure reformatting or condensing; the patcher should adapt.
  MUST_REFUSE - the new query needs content the cached answer lacks; the patcher must answer CANNOT_PATCH.
Each patched output is then graded by the small model for format compliance and for added facts.
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gistcache.config import load_config
from gistcache.patcher import PROMPT, PROMPT_V1, patch_detail
from gistcache.providers import build_provider
from gistcache.verifier.rules import extract_differences

from .spend import Tracker

HERE = Path(__file__).parent

MUST_PATCH = [
    ("How do I boil an egg?", "Place the egg in boiling water for 8 to 10 minutes. Move it to ice water for 2 minutes. Peel under running water.",
     "How do I boil an egg? Answer as a table."),
    ("What are the primary colors?", "The primary colors are red, yellow and blue. Mixing them produces the secondary colors: orange, green and purple.",
     "What are the primary colors? Answer in a bulleted list."),
    ("Why do we sleep?", "Sleep lets the brain clear waste, consolidate memories and regulate mood. It also lets the body repair tissue and balance hormones.",
     "Why do we sleep? Answer briefly."),
    ("What are the days of the week?", "The days of the week are Monday, Tuesday, Wednesday, Thursday, Friday, Saturday and Sunday.",
     "What are the days of the week? Answer as JSON."),
]
MUST_REFUSE = [
    ("How do I boil an egg?", "Place the egg in boiling water for 8 to 10 minutes. Move it to ice water for 2 minutes. Peel under running water.",
     "How do I boil an egg at high altitude, with times for 3000 metres?"),
    ("What is a mutex?", "A mutex is a lock that lets only one thread use a shared resource at a time.",
     "What is a mutex? Include a worked Rust code example."),
    ("How do I reset my router?", "Hold the reset button for 10 seconds until the lights blink, then wait for it to restart.",
     "How do I reset my router? Explain how to keep my saved settings."),
    ("What is inflation?", "Inflation is a general rise in prices over time, which reduces what each unit of money can buy.",
     "What is inflation? Compare it with deflation and give historical examples."),
]

GRADE = """You are checking an adapted answer.

Original answer:
{cached}

New query: {query}

Adapted answer:
{candidate}

1. FORMAT: does the adapted answer satisfy the format, length or constraint the new query asks for?
2. ADDED: does it contain any fact, step, number, name, example or code that is NOT in the original answer?

Reply with exactly two lines, e.g.
FORMAT: YES
ADDED: NO"""


def load_real(n_success: int):
    p = HERE / "results" / "pilot.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    pat = [r for r in rows if r["kind"] == "patchable"]
    refused = [r for r in pat if "CANNOT_PATCH" in r["reason"]]
    worked = [r for r in pat if r["tier"] == "near_hit"][:n_success]
    return [(r["seed"], None, r["query"], "refused_before" if r in refused else "worked_before") for r in refused + worked]


def grade(small, cached, query, candidate):
    out = small.complete([{"role": "user", "content": GRADE.format(cached=cached, query=query, candidate=candidate)}],
                         max_tokens=300).text.upper()
    f, a = re.search(r"FORMAT:\s*(YES|NO)", out), re.search(r"ADDED:\s*(YES|NO)", out)
    return (f.group(1) == "YES" if f else None), (a.group(1) == "YES" if a else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/anthropic.yaml")
    p.add_argument("--live", action="store_true")
    p.add_argument("--cap", type=float, default=0.08)
    p.add_argument("--max-tokens", type=int, default=300)
    a = p.parse_args()
    real = load_real(7)
    print(f"{len(real)} real cases ({sum(1 for r in real if r[3] == 'refused_before')} refused in the last pilot), "
          f"{len(MUST_PATCH)} synthetic must-patch, {len(MUST_REFUSE)} synthetic must-refuse")
    if not a.live:
        for s, _, q, tag in real:
            print(f"  [real/{tag}] {q!r}")
        for _, _, q in MUST_PATCH:
            print(f"  [must-patch] {q!r}")
        for _, _, q in MUST_REFUSE:
            print(f"  [must-refuse] {q!r}")
        return

    cfg = load_config(a.config)
    t = Tracker(a.cap, cfg["pricing"])
    large = t.wrap(build_provider(cfg["providers"]["large"]), "large")
    small = t.wrap(build_provider(cfg["providers"]["small"]), "small")
    ratio = cfg["patcher"]["max_length_ratio"]

    cpath = HERE / "data" / "patcher_cache.json"
    cache = json.loads(cpath.read_text()) if cpath.exists() else {}
    need = [s for s, *_ in real if s not in cache]
    with ThreadPoolExecutor(8) as ex:
        for s, txt in zip(need, ex.map(lambda s: large.complete([{"role": "user", "content": s}], max_tokens=a.max_tokens).text, need)):
            cache[s] = txt
    cpath.parent.mkdir(exist_ok=True)
    cpath.write_text(json.dumps(cache, indent=1))

    cases = [("real", s, cache[s], q, tag) for s, _, q, tag in real]
    cases += [("must_patch", s, ans, q, "") for s, ans, q in MUST_PATCH]
    cases += [("must_refuse", s, ans, q, "") for s, ans, q in MUST_REFUSE]

    def run(case):
        kind, seed, ans, q, tag = case
        diffs = extract_differences(seed, q)
        out = {}
        for name, prompt in (("v1", PROMPT_V1), ("v2", PROMPT)):
            text, comp, why = patch_detail(small, seed, ans, q, diffs, ratio, prompt=prompt)
            fmt = added = None
            if text:
                fmt, added = grade(small, ans, q, text)
            out[name] = {"patched": text is not None, "format_ok": fmt, "added": added, "text": text,
                         "why": why, "raw": comp.text, "cached_len": len(ans)}
        return case, out

    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(run, cases))

    print(f"\n{'':14}{'prompt':8}{'patched':>9}{'refused':>9}{'format ok':>11}{'added facts':>13}")
    for kind in ("real", "must_patch", "must_refuse"):
        rs = [o for (c, o) in results if c[0] == kind]
        for v in ("v1", "v2"):
            pat = [o[v] for o in rs if o[v]["patched"]]
            print(f"{kind:14}{v:8}{len(pat):>9}{len(rs) - len(pat):>9}"
                  f"{sum(1 for x in pat if x['format_ok']):>8}/{len(pat):<2}{sum(1 for x in pat if x['added']):>10}/{len(pat):<2}")
    print("\nregressions and notable cases:")
    for (kind, seed, ans, q, tag), o in results:
        if kind == "must_refuse" and o["v2"]["patched"]:
            print(f"  v2 PATCHED A MUST-REFUSE case: {q!r}\n      -> {o['v2']['text'][:140]!r}")
        for v in ("v1", "v2"):
            if kind != "must_refuse" and not o[v]["patched"]:
                print(f"  {v} REJECTED ({kind}) because {o[v]['why'].upper()}: {q!r}\n"
                      f"      cached answer {o[v]['cached_len']} chars; model wrote {len(o[v]['raw'])} chars: {o[v]['raw'][:150]!r}")
        if o["v2"]["patched"] and o["v2"]["added"]:
            print(f"  v2 ADDED FACTS ({kind}): {q!r}\n      -> {o['v2']['text'][:140]!r}")
        if o["v2"]["patched"] and o["v2"]["format_ok"] is False:
            print(f"  v2 FORMAT NOT MET ({kind}): {q!r}")
    print(f"\nspent ${t.total:.4f} (large {t.calls['large']} calls, small {t.calls['small']} calls)")


if __name__ == "__main__":
    main()
