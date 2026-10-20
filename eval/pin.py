"""Pin the verifier's current behaviour on every labelled eval pair, so a change cannot silently regress it.

  python -m eval.pin            # rewrite eval/pinned.json (rules + full-pipeline tiers; loads real models)
  python -m eval.pin --rules    # rewrite only the rule pins (fast, no models)

Review the git diff of eval/pinned.json before committing: every changed line is a behaviour change.
"""
import argparse
import json
from pathlib import Path

from gistcache.config import load_config
from gistcache.verifier.rules import extract_differences

HERE = Path(__file__).parent
PIN = HERE / "pinned.json"
SETS = ("hard_set.jsonl", "held_out.jsonl", "fresh.jsonl")


def load_pairs() -> list[dict]:
    rows = []
    for f in SETS:
        for line in (HERE / f).read_text().splitlines():
            r = json.loads(line)
            rows.append({**r, "key": f"{f.split('.')[0]}:{r['id']}"})
    return rows


def rule_pin(r: dict) -> list[str]:
    return sorted(f"{d.kind}|{d.severity}" for d in extract_differences(r["cached"], r["new"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rules", action="store_true", help="only rewrite the rule pins")
    a = p.parse_args()
    rows = load_pairs()
    data = json.loads(PIN.read_text()) if PIN.exists() else {}
    data["rules"] = {r["key"]: rule_pin(r) for r in rows}
    if not a.rules:
        from .run import evaluate, load_models
        cfg = load_config(None)["verifier"]
        emb, sc, guard = load_models(True, True)
        res = evaluate(rows, emb, sc, cfg["hit_threshold"], cfg["near_threshold"], 0.85, guard=guard,
                       contra_t=cfg["contradiction_threshold"])
        data["tiers"] = {r["key"]: r["got"] for r in res}
    PIN.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(f"pinned {len(rows)} pairs ({'rules only' if a.rules else 'rules + tiers'}) -> {PIN}")


if __name__ == "__main__":
    main()
