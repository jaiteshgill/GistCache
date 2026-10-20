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
    assert validate("x" * 300, cached, "q", 1.5) is None            # grew too much: probably new content
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

    assert "NOT new information" in run(None)               # v2 is the default (validated by eval.patcher_check)
    assert "WITHOUT inventing facts" in run("v1")


def test_rejections_say_why():
    from gistcache.patcher import check, patch_detail
    cached = "c" * 100
    assert check("", cached, "q", 1.5) == (None, "empty")
    assert check("CANNOT_PATCH", cached, "q", 1.5) == (None, "cannot_patch")
    assert check("x" * 300, cached, "q", 1.5) == (None, "too_long")
    assert check("x" * 10, cached, "q", 1.5) == (None, "too_short")
    assert check("x" * 120, cached, "q", 1.5) == ("x" * 120, "ok")
    text, comp, why = patch_detail(Spy("x" * 300), "a", cached, "q", D, 1.5)
    assert text is None and why == "too_long" and comp.text == "x" * 300


def test_pipeline_fallback_reason_names_the_real_cause():
    from gistcache.config import load_config
    from gistcache.embed import HashEmbedder
    from gistcache.pipeline import Pipeline
    from gistcache.store import Store
    cfg = load_config(None)
    cfg["verifier"].update(hit_threshold=0.8, near_threshold=0.5)
    p = Pipeline(cfg, Store(), HashEmbedder(), Spy("A" * 60), Spy("x" * 500))  # patcher output far too long (60-char cached answer)
    ask = lambda q: p.handle([{"role": "user", "content": q}])
    ask("sort a list of numbers in python ascending")
    r = ask("sort a list of numbers in python descending")
    assert r.tier == "miss" and r.fallback and r.reason == "miss: near_hit fallback: TOO_LONG"


# --- real patcher outputs from eval.patcher_check that the old pure-ratio rule wrongly rejected -----
def test_markup_on_a_short_answer_is_not_new_content():
    cached = "Place the egg in boiling water for 8 to 10 minutes. Move it to ice water for 2 minutes. Peel under running water."  # 113
    table = "| Step | Action | Details |\n|---|---|---|\n" + "| 1 | Boil the egg | Place the egg in boiling water for 8 to 10 minutes. |\n" * 2
    assert 1.5 * len(cached) < len(table) <= len(cached) + 150   # the old ratio rule rejected this; real run: 224 chars
    assert validate(table, cached, "how do i boil an egg? answer as a table.", 1.5) == table.strip()


def test_json_wrapper_on_a_short_answer_is_accepted():
    cached = "The days of the week are Monday, Tuesday, Wednesday, Thursday, Friday, Saturday and Sunday."  # 91
    js = '```json\n{\n  "days_of_the_week": [\n    "Monday",\n    "Tuesday",\n    "Wednesday",\n    "Thursday",\n    "Friday",\n    "Saturday",\n    "Sunday"\n  ]\n}\n```'
    assert len(js) > 1.5 * len(cached)
    assert validate(js, cached, "what are the days of the week? answer as json.", 1.5) == js


def test_a_list_may_be_much_shorter_than_prose():
    cached = "The primary colors are red, yellow and blue. Mixing them produces the secondary colors: orange, green and purple."
    assert validate("- Red\n- Yellow\n- Blue", cached, "what are the primary colors? answer in a bulleted list.", 1.5)


def test_real_inflation_of_a_long_answer_is_still_rejected():
    cached = "c" * 800
    assert validate("x" * 1300, cached, "q", 1.5) is None   # 1.6x on a long answer: probably invented content
    assert validate("x" * 1150, cached, "q", 1.5)           # within 1.5x
