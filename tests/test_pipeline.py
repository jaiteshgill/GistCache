import pytest

from gistcache.config import load_config
from gistcache.embed import HashEmbedder
from gistcache.pipeline import Pipeline
from gistcache.providers import Completion
from gistcache.store import Store


class Fake:
    def __init__(self, model, reply):
        self.model, self.reply, self.calls = model, reply, 0

    def complete(self, messages, **params):
        self.calls += 1
        r = self.reply(messages) if callable(self.reply) else self.reply
        return Completion(r, 10, 10)


def make(small_reply="patched answer that is long enough to pass validation of length ratio"):
    cfg = load_config(None)
    cfg["verifier"].update(hit_threshold=0.8, near_threshold=0.5)
    large, small = Fake("large", "A" * 60), Fake("small", small_reply)
    return Pipeline(cfg, Store(), HashEmbedder(), large, small), large, small


def ask(p, q, **kw):
    return p.handle([{"role": "user", "content": q}], **kw)


def test_miss_then_hit_on_paraphrase():
    p, large, _ = make()
    assert ask(p, "how do I reset my password").tier == "miss"
    r = ask(p, "how can I reset my password")
    assert r.tier == "hit" and large.calls == 1


def test_critical_difference_goes_to_large():
    p, large, _ = make()
    ask(p, "should I cancel my order please")
    r = ask(p, "should I not cancel my order please")
    assert r.tier == "miss" and "negation" in r.reason and large.calls == 2


def test_near_hit_uses_patcher():
    p, large, small = make()
    ask(p, "sort a list of numbers in python ascending")
    r = ask(p, "sort a list of numbers in python descending")
    assert r.tier == "near_hit" and small.calls == 1 and large.calls == 1


def test_cannot_patch_falls_back():
    p, large, small = make(small_reply="CANNOT_PATCH")
    ask(p, "sort a list of numbers in python ascending")
    r = ask(p, "sort a list of numbers in python descending")
    assert r.tier == "miss" and r.fallback and large.calls == 2


def test_scope_isolation():
    p, large, _ = make()
    ask(p, "how do I reset my password", scope="a")
    assert ask(p, "how do I reset my password", scope="b").tier == "miss"


def test_time_sensitive_never_cached():
    p, large, _ = make()
    ask(p, "what is the latest python release")
    assert ask(p, "what is the latest python release").tier == "miss"
    assert large.calls == 2


def test_bypass_paths():
    p, large, _ = make()
    assert ask(p, "hello there friend", bypass=True).tier == "bypass"
    assert ask(p, "hello there friend", temperature=1.2).tier == "bypass"
    assert ask(p, "hello there friend", has_tools=True).tier == "bypass"
    assert p.store.candidates("default", None) == []


def test_multi_turn_context_isolated():
    p, large, _ = make()
    t1 = [{"role": "user", "content": "tell me about cats"}, {"role": "assistant", "content": "cats are great"},
          {"role": "user", "content": "what do they eat"}]
    t2 = [{"role": "user", "content": "tell me about dogs"}, {"role": "assistant", "content": "dogs are great"},
          {"role": "user", "content": "what do they eat"}]
    p.handle(t1)
    assert p.handle(t2).tier == "miss"
    assert p.handle(t1).tier == "hit"


def test_decisions_logged():
    p, *_ = make()
    ask(p, "how do I reset my password")
    assert p.store.last_decisions(1)[0]["tier"] == "miss"


class FakeGuard:
    def __init__(self, value): self.value, self.calls = value, 0

    def contradiction(self, a, b):
        self.calls += 1
        return self.value


def test_contradiction_guard_vetoes_a_hit():
    p, large, _ = make()
    p.guard = FakeGuard(0.99)
    ask(p, "how do I reset my password")
    r = ask(p, "how can I reset my password")
    assert r.tier == "miss" and "contradiction" in r.reason and large.calls == 2


def test_guard_allows_a_hit_when_no_contradiction():
    p, large, _ = make()
    p.guard = FakeGuard(0.01)
    ask(p, "how do I reset my password")
    assert ask(p, "how can I reset my password").tier == "hit"


def test_guard_not_called_for_near_hits():
    p, large, small = make()
    p.guard = FakeGuard(0.99)
    ask(p, "sort a list of numbers in python ascending")
    assert ask(p, "sort a list of numbers in python descending").tier == "near_hit"
    assert p.guard.calls == 0


def test_volatility_check_skips_caching():
    p, large, small = make(small_reply="YES")
    p.cfg["cache"]["volatility_check"] = True
    ask(p, "how do I reset my password")
    assert p.store.candidates("default", None) == []
    p2, large2, small2 = make(small_reply="NO")
    p2.cfg["cache"]["volatility_check"] = True
    ask(p2, "how do I reset my password")
    assert len(p2.store.candidates("default", None)) == 1


def test_guard_covers_near_hits_that_only_the_content_net_explains():
    p, large, small = make()
    p.guard = FakeGuard(0.99)
    ask(p, "how do I structure an essay with a strong thesis")
    r = ask(p, "how do I structure an essay with a weak thesis nobody reads")  # added content, no rule explains it
    assert r.tier == "miss" and "contradiction" in r.reason


# --- borderline-hit second opinion ------------------------------------------------------------
def borderline(reply, **v):
    p, large, small = make(small_reply=reply)
    p.cfg["verifier"].update(hit_threshold=0.9, borderline_judge=True, **v)  # "change" vs "reset" scores ~0.83
    return p, large, small


def test_borderline_judge_yes_promotes_to_hit_and_counts_cost():
    p, large, small = borderline("YES\nSame question.")
    p.cfg["pricing"] = {"small": {"input": 1000.0, "output": 1000.0}}
    ask(p, "how do i reset my password")
    r = ask(p, "how do i change my password")
    assert r.tier == "hit" and r.reason == "hit: confirmed by judge"
    assert large.calls == 1 and small.calls == 1 and r.cost_usd > 0


def test_borderline_judge_no_stays_a_miss():
    p, large, small = borderline("NO\nDifferent action.")
    ask(p, "how do i reset my password")
    r = ask(p, "how do i change my password")
    assert r.tier == "miss" and large.calls == 2 and small.calls == 1


def test_borderline_judge_is_off_by_default():
    p, large, small = make(small_reply="YES")
    p.cfg["verifier"]["hit_threshold"] = 0.9
    ask(p, "how do i reset my password")
    assert ask(p, "how do i change my password").tier == "miss" and small.calls == 0


def test_borderline_judge_never_overrides_a_rule():
    p, large, small = borderline("YES")
    ask(p, "should i cancel my order please")
    assert ask(p, "should i not cancel my order please").tier == "miss" and small.calls == 0


def test_borderline_judge_skips_unrelated_queries():
    p, large, small = borderline("YES")
    ask(p, "how do i reset my password")
    assert ask(p, "what is the capital of france").tier == "miss" and small.calls == 0


def test_contradiction_guard_still_vetoes_a_judge_promoted_hit():
    p, large, small = borderline("YES")
    p.guard = FakeGuard(0.99)
    ask(p, "how do i reset my password")
    r = ask(p, "how do i change my password")
    assert r.tier == "miss" and "contradiction" in r.reason
    assert small.calls == 0  # the free local check runs first, so no paid judge call is wasted


def test_borderline_verdict_parsing():
    from gistcache.verifier.borderline import parse
    assert parse("YES. same") and parse("**YES**") and not parse("NO. differs") and not parse("") and not parse("unsure")
