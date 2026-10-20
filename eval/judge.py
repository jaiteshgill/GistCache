"""LLM judge: is a served answer acceptable for the query, judged against a fresh reference answer."""
import re

PROMPT = """You are grading an answer served from a cache.

Query: {query}

Reference answer (written fresh for this exact query):
{reference}

Candidate answer (served from cache or adapted from a cached answer):
{candidate}

Is the candidate an acceptable answer to THIS query? It must answer what was actually asked (including any
requested format, order, length or constraint), contain no facts that are wrong for this query, and not
answer a different question. Minor wording differences from the reference are fine. Both answers may end
abruptly because of a length limit: ignore truncation and compare only the content that is present.
Reply on the first line with exactly YES or NO, then one short sentence explaining why."""


def verdict(text: str) -> bool | None:
    """First YES/NO in the reply, tolerant of markdown like **YES** or 'Verdict: NO'. None if neither."""
    m = re.search(r"\b(YES|NO)\b", text.upper())
    return None if not m else m.group(1) == "YES"


# max_tokens is generous on purpose: the model can spend part of the budget on hidden reasoning, and a
# too-small limit returned empty or cut-off replies that an earlier version misread as "NO".
def judge(small, query: str, reference: str, candidate: str) -> tuple[bool | None, str]:
    out = small.complete([{"role": "user", "content": PROMPT.format(
        query=query, reference=reference, candidate=candidate)}], max_tokens=600).text
    return verdict(out), out.strip()
