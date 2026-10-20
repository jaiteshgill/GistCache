"""Second opinion on borderline hits.

The cross-encoder says "probably the same question" above hit_threshold and "different" far below it. In
between it is unsure, and it rejects many true paraphrases there. For that band only, ask the small model
whether one answer would serve both queries. Rules and the contradiction guard still run afterwards, so this
can only promote a pair that every other check has already left clear.
"""
import re

PROMPT = """Two users asked a question. Would ONE answer be fully correct and complete for BOTH of them?

Reply NO if the questions differ in any detail that changes the answer: a different person, place, product,
number, date, version, language, condition, scope, or a requested format or length. Reply YES only if they
are the same question in different words.

Question A: {a}
Question B: {b}

Reply on the first line with exactly YES or NO, then one short sentence of reasoning."""


def parse(text: str) -> bool:
    m = re.search(r"\b(YES|NO)\b", text.upper())
    return bool(m) and m.group(1) == "YES"  # no verdict, or an empty reply, counts as NO


def same_answer(small, a: str, b: str):
    # max_tokens is generous: the model may spend part of the budget on hidden reasoning, and a small
    # limit produced empty replies in an earlier judge.
    c = small.complete([{"role": "user", "content": PROMPT.format(a=a, b=b)}], max_tokens=600)
    return parse(c.text), c
