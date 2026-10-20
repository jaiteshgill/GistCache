"""Does a small-model second opinion recover true paraphrases from the borderline band without letting
wrong ones through?

  python -m eval.borderline                 # free: list the pairs that would reach the judge
  python -m eval.borderline --live --cap 0.03   # asks the small model about each (needs an API key; ~$0.01)

Only pairs the rules find clean and the contradiction guard clears, whose scorer score is in
[borderline_low, hit_threshold), are asked. Pairs above the threshold already hit, pairs below never will.
"""
import argparse
import json
from pathlib import Path

from gistcache.config import load_config
from gistcache.normalize import normalize
from gistcache.providers import build_provider
from gistcache.verifier.borderline import same_answer
from gistcache.verifier.rules import extract_differences

from .pin import load_pairs
from .run import load_models
from .spend import Tracker

HERE = Path(__file__).parent


def pairs():
    rows = [{"cached": r["cached"], "new": r["new"], "exp": r["expected"], "src": r["key"]} for r in load_pairs()]
    vp = HERE / "data" / "variants.json"
    if vp.exists():  # local pilot variants (gitignored); a paraphrase must hit, the others must not
        for s, d in json.loads(vp.read_text()).items():
            for k, e in (("paraphrase", "hit"), ("patchable", "near_hit"), ("critical", "miss")):
                rows.append({"cached": s, "new": d[k], "exp": e, "src": f"pilot:{k}"})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/anthropic.yaml")
    p.add_argument("--live", action="store_true")
    p.add_argument("--cap", type=float, default=0.03)
    a = p.parse_args()
    cfg = load_config(a.config)
    v = cfg["verifier"]
    emb, sc, guard = load_models(True, True)

    band = []
    for r in pairs():
        if r["src"].startswith("fresh:volatile"):
            continue  # identical text; the volatility check owns these, not the judge
        if extract_differences(r["cached"], r["new"]):
            continue
        cos = float(emb.encode(normalize(r["cached"])) @ emb.encode(normalize(r["new"])))
        score = sc.score(r["cached"], r["new"], cos)
        if cos >= v["near_threshold"] and v["borderline_low"] <= score < v["hit_threshold"] \
                and guard.contradiction(r["cached"], r["new"]) < v["contradiction_threshold"]:
            band.append({**r, "score": score})
    pos = [r for r in band if r["exp"] == "hit"]
    print(f"{len(band)} pairs reach the judge: {len(pos)} true paraphrases, {len(band) - len(pos)} that must NOT hit")
    if not a.live:
        for r in band:
            print(f"  [{r['exp']:8}] score={r['score']:.2f}  {r['cached']!r} -> {r['new']!r}")
        print("\n(free listing; add --live to ask the small model)")
        return

    t = Tracker(a.cap, cfg["pricing"])
    small = t.wrap(build_provider(cfg["providers"]["small"]), "small")
    recovered = wrong = 0
    for r in band:
        ok, c = same_answer(small, r["cached"], r["new"])
        good = ok == (r["exp"] == "hit")
        recovered += ok and r["exp"] == "hit"
        wrong += ok and r["exp"] != "hit"
        print(f"  {'ok ' if good else 'BAD'} judge={'YES' if ok else 'NO '} expected={r['exp']:8} {r['cached']!r} -> {r['new']!r}\n"
              f"        {c.text.strip()[:110]!r}")
    print(f"\nrecovered {recovered}/{len(pos)} true paraphrases; wrongly promoted {wrong}/{len(band) - len(pos)}; spent ${t.total:.4f}")


if __name__ == "__main__":
    main()
