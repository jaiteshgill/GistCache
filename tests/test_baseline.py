from eval.baseline import Memo, ThresholdCache
from gistcache.embed import HashEmbedder
from gistcache.providers import Completion


class Counting:
    model = "large"

    def __init__(self):
        self.calls = 0

    def complete(self, messages, **params):
        self.calls += 1
        return Completion(f"answer to {messages[-1]['content']}", 10, 100)


def test_memo_answers_each_request_once_and_replays_the_same_text():
    inner = Counting()
    m = Memo(inner)
    a = m.complete([{"role": "user", "content": "q"}], max_tokens=5)
    b = m.complete([{"role": "user", "content": "q"}], max_tokens=5)
    c = m.complete([{"role": "user", "content": "q"}], max_tokens=6)   # different params: a different request
    assert inner.calls == 2 and a is b and c is not a and len(m.latencies()) == 2


def test_threshold_cache_hits_above_threshold_and_charges_only_misses():
    pricing = {"large": {"input": 1_000_000.0, "output": 0.0}}
    tc = ThresholdCache(HashEmbedder(), Memo(Counting()), 0.8, pricing)
    r1 = tc.handle("how do i reset my password", 5)
    r2 = tc.handle("how can i reset my password", 5)        # cosine ~0.87
    r3 = tc.handle("what is the capital of france", 5)
    assert (r1["tier"], r2["tier"], r3["tier"]) == ("miss", "hit", "miss")
    assert r2["text"] == "answer to how do i reset my password" and r2["cost"] == 0.0 and r1["cost"] == 10.0


def test_threshold_cache_reuses_a_wrong_answer_on_a_lookalike():
    # the failure GistCache exists to prevent: negation barely moves cosine
    tc = ThresholdCache(HashEmbedder(), Memo(Counting()), 0.8, {})
    tc.handle("should i cancel my order", 5)
    assert tc.handle("should i not cancel my order", 5)["tier"] == "hit"
