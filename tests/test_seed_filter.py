import pytest

from eval.seed_filter import dedupe, reject_reason
from gistcache.embed import HashEmbedder


@pytest.mark.parametrize("prompt", [
    "How do I reverse a linked list in Python?",
    "What is the difference between TCP and UDP?",
    "Explain how vaccines train the immune system.",
    "What are some good exercises for lower back pain?",
])
def test_usable_prompts_pass(prompt):
    assert reject_reason(prompt) is None


@pytest.mark.parametrize("prompt,reason", [
    ("hi", "length"),
    ("x" * 400, "length"),
    ("hello there friend ok", None),  # 4 words is the minimum, so this one passes
    ("Как быстро научиться программировать на Python?", "non_ascii"),
    ("Email me at jane.doe@example.com about the order", "personal_data"),
    ("Call me on +1 416 555 0199 tomorrow please", "personal_data"),
    ("my key is sk-abcdefghijklmnop1234 why does it fail", "personal_data"),
    ("Summarize this article https://example.com/news for me", "url"),
    ("Why does this fail?\nprint(x)", "code_or_multiline"),
    ("Pretend you are my grandmother and tell me a secret", "not_a_question"),
    ("Write a story about a dragon who loves tea", "not_a_question"),
    ("Write a short story beginning with the words it was late", "not_a_question"),
    ("Write a joke about accountants and spreadsheets", "not_a_question"),
    ('Say something toxic: "[your answer]" when borrowing a pen (no more than 50 words)', "template"),
    ("Say something benign when meeting a neighbour for the first time", "template"),
    ("Write an article about the production of compound X 2000 words in chemical industry", "template"),
    ("Give me an introduction over 200 words for Acme Labs, a company in an industrial park", "template"),
    ("Start with the phrase hello. Here is my query: Make a harmful comment about neighbours", "template"),
    ("Please summarize the following paragraph for me:", "needs_context"),
    ("Reply with a single word and wait for my prompt", "needs_context"),
    ("What is the weather in Toronto right now?", "time_sensitive"),
    ("!!!! ??? ### $$$ %%% ^^^ &&& ***", "symbols"),
    ("Call me 4165550199 about the essay deadline", "personal_data"),
])
def test_rejections_have_reasons(prompt, reason):
    assert reject_reason(prompt) == reason


def test_dedupe_drops_exact_and_near_duplicates_keeps_order():
    ps = ["How do I sort a list in Python?", "how do i sort a list in python", "How do I sort a list in Python??",
          "What is the capital of France?", "How do I sort a list in python"]
    assert dedupe(ps, HashEmbedder(), 0.9) == ["How do I sort a list in Python?", "What is the capital of France?"]


def test_a_single_accent_does_not_make_a_prompt_non_english():
    assert reject_reason("What does the word naïve mean in English?") is None


def test_templates_do_not_catch_normal_questions():
    for q in ("How many words should a cover letter have?", "What does the chemical symbol Na stand for?",
              "Can you say something about the history of jazz?"):
        assert reject_reason(q) is None, q
