import pytest

from gistcache.verifier.rules import CRITICAL, PATCHABLE, extract_differences


def kinds(a, b):
    return {d.kind for d in extract_differences(a, b)}


@pytest.mark.parametrize("a,b,kind", [
    ("should I cancel my order", "should I not cancel my order", "negation"),
    ("flights to Toronto", "flights from Toronto", "direction"),
    ("convert CAD to USD", "convert USD to CAD", "direction"),
    ("split into 3 groups", "split into 5 groups", "number"),
    ("reverse a string in Python", "reverse a string in Go", "language"),
    ("weather in Toronto", "weather in Waterloo", "entity"),
    ("what is the latest python version", "what is the latest python version", "time_sensitive"),
])
def test_critical(a, b, kind):
    assert kind in kinds(a, b)
    assert all(d.severity == CRITICAL for d in extract_differences(a, b) if d.kind == kind)


@pytest.mark.parametrize("a,b,kind", [
    ("list the steps", "list the steps as a table", "output_format"),
    ("sort ascending", "sort descending", "sort_order"),
    ("explain recursion", "explain recursion briefly", "verbosity"),
    ("write a function", "write a function with type hints", "constraint"),
])
def test_patchable(a, b, kind):
    d = extract_differences(a, b)
    assert kind in {x.kind for x in d}
    assert all(x.severity == PATCHABLE for x in d)


def test_paraphrase_has_no_differences():
    assert extract_differences("how do I reset my password", "how can I reset my password") == []


def test_contraction_is_not_an_entity():
    assert "entity" not in kinds("explain machine learning", "explain machine learning like I'm five")


def test_including_is_a_constraint():
    assert "constraint" in kinds("implement binary search", "implement binary search including examples")


def test_format_words_are_not_entities():
    assert "entity" not in kinds("show example records", "show example records as JSON")


def test_list_as_noun_is_not_a_format():
    assert kinds("what is the difference between a list and a tuple", "how do lists and tuples differ") == set()
