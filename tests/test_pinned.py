"""Regression pins over every labelled eval pair.

The rule pins are fast and always run. The tier pins load the real embedder, cross-encoder and NLI guard
(~10 s), so they only run when GISTCACHE_SLOW=1. If a change is intended, run `python -m eval.pin`,
review the diff of eval/pinned.json, and commit it with the change.
"""
import json
import os

import pytest

from eval.pin import PIN, load_pairs, rule_pin

PAIRS = load_pairs()
PINNED = json.loads(PIN.read_text())


def test_every_pair_is_pinned():
    keys = {r["key"] for r in PAIRS}
    assert keys == set(PINNED["rules"]), "eval sets changed: run `python -m eval.pin` and review the diff"


@pytest.mark.parametrize("r", PAIRS, ids=[r["key"] for r in PAIRS])
def test_rule_differences_are_unchanged(r):
    assert rule_pin(r) == PINNED["rules"][r["key"]], (
        f"rule output changed for {r['cached']!r} -> {r['new']!r}; if intended, run `python -m eval.pin` and review")


@pytest.mark.skipif(os.environ.get("GISTCACHE_SLOW") != "1", reason="loads real models; set GISTCACHE_SLOW=1")
def test_full_pipeline_tiers_are_unchanged():
    from eval.run import evaluate, load_models
    from gistcache.config import load_config
    cfg = load_config(None)["verifier"]
    emb, sc, guard = load_models(True, True)
    res = evaluate(PAIRS, emb, sc, cfg["hit_threshold"], cfg["near_threshold"], 0.85, guard=guard,
                   contra_t=cfg["contradiction_threshold"])
    changed = [(r["key"], PINNED["tiers"][r["key"]], r["got"]) for r in res if r["got"] != PINNED["tiers"][r["key"]]]
    assert not changed, f"{len(changed)} tier decisions changed (key, pinned, now): {changed[:8]}"
