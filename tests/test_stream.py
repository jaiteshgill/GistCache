import random

from eval.spend import BudgetExceeded, Tracker
from eval.stream import FakeProvider, build_queries
from eval.variants import parse

import pytest


def test_tracker_stops_at_cap():
    t = Tracker(0.001, {"large": {"input": 2.0, "output": 10.0}})
    p = t.wrap(FakeProvider("large"), "large")
    with pytest.raises(BudgetExceeded):
        for _ in range(100):
            p.complete([{"role": "user", "content": "hello there " * 50}])
    assert t.total >= 0.001 and t.calls["large"] >= 1


def test_every_seed_is_asked_before_its_variants():
    variants = {f"seed {i}": {"paraphrase": "p", "patchable": "q", "critical": "r"} for i in range(5)}
    qs = build_queries(variants, random.Random(1))
    assert [q["kind"] for q in qs[:5]] == ["seed"] * 5 and len(qs) == 20


def test_parse_variants_rejects_malformed():
    assert parse('noise {"paraphrase":"a","patchable":"b","critical":"c"} tail') == {"paraphrase": "a", "patchable": "b", "critical": "c"}
    assert parse('{"paraphrase":"a"}') is None
    assert parse("no json") is None
