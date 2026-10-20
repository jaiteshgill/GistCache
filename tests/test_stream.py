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


def test_judge_verdict_parsing():
    from eval.judge import verdict
    assert verdict("YES\nSame content.") is True
    assert verdict("**YES**") is True
    assert verdict("Verdict: NO - wrong format") is False
    assert verdict("no") is False
    assert verdict("maybe") is None


def test_judge_gives_the_model_room_and_treats_empty_replies_as_unknown():
    from eval.judge import judge
    seen = {}

    class P:
        def complete(self, messages, **params):
            seen.update(params)
            from gistcache.providers import Completion
            return Completion("")

    ok, why = judge(P(), "q", "ref", "cand")
    assert ok is None and why == "" and seen["max_tokens"] >= 400


def test_published_report_never_contains_prompt_text_but_local_copy_does():
    from eval.spend import Tracker
    from eval.stream import report
    secret = "PRIVATE-USER-PROMPT about my landlord"
    rows = [{"kind": "paraphrase", "query": secret, "tier": "hit", "reason": "hit: paraphrase", "cost": 0.0,
             "ms": 100.0, "ok": False, "why": "NO: answers a different question"},
            {"kind": "seed", "query": "another private seed", "tier": "miss", "reason": "miss: no candidates",
             "cost": 0.003, "ms": 4000.0}]
    base = {0.8: [{"kind": "paraphrase", "query": secret, "tier": "hit", "cost": 0.0, "ms": 5.0, "ok": False}]}
    args = (rows, base, Tracker(1.0, {}), 0.003, [4000.0], False, 0, "t")
    published, local = report(*args, True), report(*args, False)
    assert secret not in published and "another private seed" not in published
    assert "NOT acceptable [hit, paraphrase]" in published and "may not be redistributed" in published
    assert secret in local


def test_end_to_end_table_counts_wrong_reuse_per_system():
    from eval.spend import Tracker
    from eval.stream import report
    rows = [{"kind": "paraphrase", "query": "q", "tier": "hit", "reason": "r", "cost": 0.0, "ms": 1.0, "ok": True}]
    base = {0.8: [{"kind": "critical", "query": "q", "tier": "hit", "cost": 0.0, "ms": 1.0, "ok": False},
                  {"kind": "seed", "query": "s", "tier": "miss", "cost": 0.003, "ms": 4000.0}]}
    md = report(rows, base, Tracker(1.0, {}), 0.003, [4000.0], False, 0, "t", False)
    assert "| Threshold cache (cosine >= 0.8) | 50% | 0/1 | 1 (50.0 per 100) |" in md
    assert "| **GistCache** | 100% | 1/1 | 0 (0.0 per 100) |" in md
    assert "| 0.8 | 0/0 | 0/0 | 1/1 |" in md  # the critical-variant hit is called out


def test_a_failing_query_is_skipped_not_fatal(monkeypatch, tmp_path):
    import sys
    import eval.stream as st
    real = st.FakeProvider.complete
    state = {"n": 0}

    def flaky(self, messages, **params):
        text = messages[-1]["content"]
        if "Rust" in text and "building a test set" not in text and "grading" not in text:
            raise RuntimeError("upstream kept failing")  # every critical variant fails permanently
        return real(self, messages, **params)

    monkeypatch.setattr(st.FakeProvider, "complete", flaky)
    monkeypatch.setattr(st, "HERE", tmp_path)
    monkeypatch.setattr(sys, "argv", ["stream", "--dry-run", "--n", "3", "--cap", "5", "--baselines", "0.8"])
    st.main()
    md = (tmp_path / "results" / "pilot_dry.md").read_text()
    assert "requests failed after retries and were skipped" in md
    assert "| seed | 3 |" in md and "| critical |" not in md  # the run finished; the failed queries are excluded


def test_variant_generation_gives_the_model_room():
    from eval import variants
    seen = {}

    class P:
        def complete(self, messages, **params):
            seen.update(params)
            from gistcache.providers import Completion
            return Completion('{"paraphrase":"a","patchable":"b","critical":"c"}')

    assert variants._one(P(), "seed") == {"paraphrase": "a", "patchable": "b", "critical": "c"}
    assert seen["max_tokens"] >= 1000
