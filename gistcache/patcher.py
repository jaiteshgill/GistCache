from .providers import Completion, Provider

CANNOT_PATCH = "CANNOT_PATCH"

PROMPT = """You adapt a cached answer to a new query WITHOUT inventing facts.
Rules:
1. Preserve everything in the cached answer that is still valid for the new query.
2. Change only what the listed differences require.
3. Do not add facts that are not in the cached answer.
4. If the new query needs information the cached answer does not contain, reply only with {cp}.

Original query: {old_q}
Cached answer:
{answer}

New query: {new_q}
Differences: {diffs}"""


def patch(provider: Provider, old_q: str, answer: str, new_q: str, diffs: list, max_ratio: float
          ) -> tuple[str | None, Completion | None]:
    """Return (patched text or None if rejected, raw completion for cost accounting)."""
    prompt = PROMPT.format(cp=CANNOT_PATCH, old_q=old_q, answer=answer, new_q=new_q,
                           diffs="; ".join(str(d) for d in diffs))
    c = provider.complete([{"role": "user", "content": prompt}])
    return (validate(c.text, answer, new_q, max_ratio), c)


def validate(text: str, cached: str, new_q: str, max_ratio: float) -> str | None:
    t = text.strip()
    if not t or CANNOT_PATCH in t:
        return None
    if len(t) > max_ratio * max(len(cached), 1):
        return None
    brief = any(w in new_q.lower() for w in ("brief", "short", "concise", "tl;dr", "eli5"))
    if len(t) < 0.2 * len(cached) and not brief:
        return None
    return t
