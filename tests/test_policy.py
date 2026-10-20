from gistcache.verifier.policy import decide
from gistcache.verifier.rules import CRITICAL, PATCHABLE, Difference

C = Difference("negation", CRITICAL, "negation mismatch")
P = Difference("sort_order", PATCHABLE, "asc -> desc")


def test_critical_wins_even_with_high_score():
    assert decide([C], 0.99, 0.9, 0.7).tier == "miss"


def test_hit():
    assert decide([], 0.95, 0.9, 0.7).tier == "hit"


def test_near_hit():
    d = decide([P], 0.8, 0.9, 0.7)
    assert d.tier == "near_hit" and d.reason == "near_hit: sort order"


def test_low_score_miss():
    assert decide([P], 0.5, 0.9, 0.7).tier == "miss"
    assert decide([], 0.8, 0.9, 0.7).tier == "miss"
