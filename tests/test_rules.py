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


def test_prefix_negation():
    assert "negation" in kinds("is it legal to record calls", "is it illegal to record calls")


def test_antonym_lexicon_uses_stems():
    assert "negation" in kinds("how do I allow cookies", "how do I blocking cookies")
    assert "negation" in kinds("how to speed up my laptop", "how to slow down my laptop")


def test_contractions_normalize_to_the_same_negation():
    assert "negation" not in kinds("my laptop wont turn on", "my laptop does not turn on")
    assert "negation" in kinds("why does my code work", "why doesn't my code work")


def test_no_abbreviation_is_not_negation():
    assert "negation" not in kinds("what is the no. of items", "what is the number of items")


def test_number_words_and_ordinals():
    assert "number" in kinds("give me three reasons", "give me five reasons")
    assert "number" in kinds("explain the 1st amendment", "explain the 2nd amendment")
    assert "number" in kinds("dress for 10 degrees", "dress for minus 10 degrees")


def test_sort_phrases_do_not_count_as_numbers():
    d = {x.kind for x in extract_differences("sort users by date", "sort users by date newest first")}
    assert "number" not in d and "sort_order" in d


def test_ambiguous_language_names_need_context():
    assert "language" not in kinds("go to the airport", "get to the airport")
    assert "language" in kinds("write a function in c", "write a function in c++")
    assert "language" in kinds("write a function in go", "write a function in rust")


def test_symmetric_comparison_is_not_a_direction_change():
    assert "direction" not in kinds("difference between mean and median", "difference between median and mean")


def test_content_added_blocks_a_plain_hit():
    assert "content_added" in kinds("what is a mutex", "what is a mutex with locking semantics")
    assert extract_differences("what is a mutex", "what is a mutex with locking semantics", strict_content=False) == []


def test_volatile_phrases():
    for q in ("weather in toronto", "who is the ceo of apple", "what time is it", "is the store open"):
        assert "time_sensitive" in kinds(q, q)


# --- regressions found by the pilot on real LLM-written text ---------------------------------
def test_second_sentence_start_is_not_an_entity():
    assert "entity" not in kinds("How do I improve my credit score?", "How do I improve my credit score? Answer briefly, in a bulleted list.")
    assert "entity" not in kinds("How do I write a cover letter?", "How do I write a cover letter? Please answer briefly.")


def test_possessive_does_not_drop_a_language():
    assert "language" not in kinds("How does garbage collection work in Java?", "Can you explain how Java's garbage collector operates?")


def test_non_and_no_are_the_same_negation():
    assert "negation" not in kinds("explain an API to a non-technical person", "explain an API to someone with no technical background")


def test_weather_as_a_topic_is_not_time_sensitive():
    assert "time_sensitive" not in kinds("difference between weather and climate", "how do weather and climate differ")
    assert "time_sensitive" in kinds("weather in toronto", "weather in toronto")


def test_to_from_in_different_phrasings_is_not_a_direction_swap():
    assert "direction" not in kinds("How do I remove duplicates from a list in JavaScript?",
                                    "What is the way to get rid of repeated elements in an array in JavaScript?")
    assert "direction" in kinds("flights to Toronto", "flights from Toronto")


def test_type_name_capitalization_is_not_a_new_entity():
    assert "entity" not in kinds("How do I convert a string to an integer in Java?", "What is the way to turn a String into an int in Java?")


# --- a content swap hiding behind a patchable difference (found by the pilot) -----------------
def test_swap_hidden_behind_a_patchable_difference_is_marked():
    d = {x.kind for x in extract_differences("Give me a short bedtime story about a brave rabbit",
                                             "Give me a brief bedtime story about a cowardly rabbit")}
    assert "content_substituted" in d


def test_genuine_patchable_pairs_are_not_marked_as_substitutions():
    for a, b in [("sort a list of numbers ascending", "sort a list of numbers descending"),
                 ("explain recursion", "explain recursion briefly"),
                 ("list the steps to bake bread", "list the steps to bake bread as a table")]:
        assert "content_substituted" not in {x.kind for x in extract_differences(a, b)}


def test_plain_paraphrase_substitution_does_not_become_a_near_hit():
    # no other patchable difference, so word swaps stay with the scorer (hit) and the guard
    assert extract_differences("how do i delete a branch", "how can i remove a branch") == []
