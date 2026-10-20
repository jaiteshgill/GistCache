import random

from eval.seeds_build import dolly_prompt, lmsys_prompt, select
from gistcache.embed import HashEmbedder


def lm(text, **kw):
    row = {"conversation": [{"role": "user", "content": text}, {"role": "assistant", "content": "a"}],
           "turn": 1, "language": "English", "redacted": False, "openai_moderation": [{"flagged": False}]}
    row.update(kw)
    return row


def test_lmsys_extraction_rules():
    assert lmsys_prompt(lm("How do I learn Python quickly?")) == ("How do I learn Python quickly?", None)
    assert lmsys_prompt(lm("x", language="Portuguese"))[1] == "not_english"
    assert lmsys_prompt(lm("x", redacted=True))[1] == "redacted"
    assert lmsys_prompt(lm("x", openai_moderation=[{"flagged": False}, {"flagged": True}]))[1] == "moderation_flagged"
    two = lm("x", turn=2)
    two["conversation"] += [{"role": "user", "content": "and then?"}]
    assert lmsys_prompt(two)[1] == "multi_turn"
    assert lmsys_prompt({})[1] == "multi_turn"  # missing fields never crash


def test_dolly_extraction_rules():
    assert dolly_prompt({"instruction": "Why is the sky blue?", "context": "", "category": "open_qa"})[1] is None
    assert dolly_prompt({"instruction": "Summarize", "context": "passage", "category": "summarization"})[1] == "needs_context"
    assert dolly_prompt({"instruction": "Write a poem", "context": "", "category": "creative_writing"})[1] == "category"


def test_select_filters_dedupes_samples_and_counts_reasons():
    rows = [lm("How do I learn Python quickly?"), lm("how do i learn python quickly"), lm("hi"),
            lm("What is the capital of Peru?", language="Spanish"), lm("Why do cats purr so much?"),
            lm("What causes ocean tides to happen?"), lm("Email me at a@b.com about the tides please")]
    seeds, stats = select(rows, lmsys_prompt, n=2, embedder=HashEmbedder(), rng=random.Random(1), max_scan=100)
    assert len(seeds) == 2 and set(seeds) <= {"How do I learn Python quickly?", "Why do cats purr so much?",
                                               "What causes ocean tides to happen?"}
    assert stats["scanned"] == 7 and stats["unique"] == 3
    assert stats["rejected"] == {"length": 1, "not_english": 1, "personal_data": 1, "near_duplicate": 1}


def test_select_stops_at_max_scan():
    rows = (lm(f"What is fact number {i} about the ocean?") for i in range(10_000))
    _, stats = select(rows, lmsys_prompt, n=1000, embedder=HashEmbedder(), rng=random.Random(1), max_scan=50)
    assert stats["scanned"] == 50
