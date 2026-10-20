from gistcache.patcher import CANNOT_PATCH, PROMPT, PROMPT_V1, patch, validate
from gistcache.providers import Completion
from gistcache.verifier.rules import Difference, PATCHABLE


class Spy:
    model = "small"

    def __init__(self, reply="adapted"):
        self.reply, self.prompts = reply, []

    def complete(self, messages, **params):
        self.prompts.append(messages[0]["content"])
        return Completion(self.reply, 5, 5)


D = [Difference("output_format", PATCHABLE, "[] -> ['table']")]


def test_prompt_contains_the_inputs_and_the_refusal_token():
    spy = Spy("x" * 40)
    patch(spy, "old q", "cached text " * 5, "new q as a table", D, 1.5)
    p = spy.prompts[0]
    assert "old q" in p and "new q as a table" in p and "cached text" in p and "output_format" in p
    assert CANNOT_PATCH in p


def test_new_prompt_says_reformatting_is_not_new_information():
    assert "NOT new information" in PROMPT and "table" in PROMPT and "bulleted" in PROMPT
    assert "NOT new information" not in PROMPT_V1  # v1 is the strict baseline the A/B check compares against


def test_prompt_can_be_swapped_for_comparison():
    spy = Spy()
    patch(spy, "a", "b" * 40, "c", D, 1.5, prompt=PROMPT_V1)
    assert "WITHOUT inventing facts" in spy.prompts[0]


def test_refusal_is_rejected_even_with_surrounding_text():
    assert validate("CANNOT_PATCH", "c" * 100, "q", 1.5) is None
    assert validate("Sorry. CANNOT_PATCH.", "c" * 100, "q", 1.5) is None


def test_validation_rules():
    cached = "c" * 100
    assert validate("", cached, "q", 1.5) is None
    assert validate("x" * 200, cached, "q", 1.5) is None            # grew too much: probably new content
    assert validate("x" * 10, cached, "explain it", 1.5) is None     # collapsed with no reason
    assert validate("x" * 10, cached, "explain it briefly", 1.5) == "x" * 10
    assert validate("x" * 120, cached, "q", 1.5) == "x" * 120


def test_pipeline_uses_the_configured_patcher_prompt():
    from gistcache.config import load_config
    from gistcache.embed import HashEmbedder
    from gistcache.pipeline import Pipeline
    from gistcache.store import Store

    def run(prompt):
        cfg = load_config(None)
        cfg["verifier"].update(hit_threshold=0.8, near_threshold=0.5)
        if prompt:
            cfg["patcher"]["prompt"] = prompt
        spy = Spy("x" * 60)
        large = Spy("A" * 60)
        p = Pipeline(cfg, Store(), HashEmbedder(), large, spy)
        ask = lambda q: p.handle([{"role": "user", "content": q}])
        ask("sort a list of numbers in python ascending")
        assert ask("sort a list of numbers in python descending").tier == "near_hit"
        return spy.prompts[0]

    assert "WITHOUT inventing facts" in run(None)          # default stays v1 until v2 is validated
    assert "NOT new information" in run("v2")


def test_rejections_say_why():
    from gistcache.patcher import check, patch_detail
    cached = "c" * 100
    assert check("", cached, "q", 1.5) == (None, "empty")
    assert check("CANNOT_PATCH", cached, "q", 1.5) == (None, "cannot_patch")
    assert check("x" * 200, cached, "q", 1.5) == (None, "too_long")
    assert check("x" * 10, cached, "q", 1.5) == (None, "too_short")
    assert check("x" * 120, cached, "q", 1.5) == ("x" * 120, "ok")
    text, comp, why = patch_detail(Spy("x" * 200), "a", cached, "q", D, 1.5)
    assert text is None and why == "too_long" and comp.text == "x" * 200


def test_pipeline_fallback_reason_names_the_real_cause():
    from gistcache.config import load_config
    from gistcache.embed import HashEmbedder
    from gistcache.pipeline import Pipeline
    from gistcache.store import Store
    cfg = load_config(None)
    cfg["verifier"].update(hit_threshold=0.8, near_threshold=0.5)
    p = Pipeline(cfg, Store(), HashEmbedder(), Spy("A" * 60), Spy("x" * 500))  # patcher output far too long
    ask = lambda q: p.handle([{"role": "user", "content": q}])
    ask("sort a list of numbers in python ascending")
    r = ask("sort a list of numbers in python descending")
    assert r.tier == "miss" and r.fallback and r.reason == "miss: near_hit fallback: TOO_LONG"
