"""LLM judge: is a served answer acceptable for the query, judged against a fresh reference answer."""

PROMPT = """You are grading an answer served from a cache.

Query: {query}

Reference answer (written fresh for this exact query):
{reference}

Candidate answer (served from cache or adapted from a cached answer):
{candidate}

Is the candidate an acceptable answer to THIS query? It must answer what was actually asked (including any
requested format, order, length or constraint), contain no facts that are wrong for this query, and not
answer a different question. Minor wording differences from the reference are fine.
Reply with only YES or NO."""


def judge(small, query: str, reference: str, candidate: str) -> bool:
    out = small.complete([{"role": "user", "content": PROMPT.format(
        query=query, reference=reference, candidate=candidate)}], max_tokens=5).text
    return out.strip().upper().startswith("YES")
