from .providers import Completion, Provider

CANNOT_PATCH = "CANNOT_PATCH"

# v1 refused too often: in a pilot it answered CANNOT_PATCH to pure reformatting requests ("in a short table",
# "as a bulleted list"), which need no new information. Kept so eval/patcher_check.py can compare the two.
PROMPT_V1 = """You adapt a cached answer to a new query WITHOUT inventing facts.
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

PROMPT = """You adapt a cached answer so that it answers a new query. The new query differs from the cached one
only in the ways listed under "Differences".

You MAY (none of this counts as inventing facts):
- Restructure what is already there: turn prose into a table, a bulleted or numbered list, JSON or a paragraph;
  reorder items; change the sort order.
- Condense: shorten, summarize or keep only the key points when asked to be brief or concise; simplify the
  wording or tone using only the ideas already present.
- Apply a narrowing constraint by selecting the part of the cached answer that satisfies it.

You MUST NOT:
- Add facts, steps, numbers, names, examples or code that are not in the cached answer.
- Change the meaning of anything you keep.

Reply with exactly {cp} (and nothing else) only if a correct answer would need content the cached answer does not
contain, such as a new example, a new comparison, or extra detail. A request to present the existing content
differently is NOT new information, so do that.

Output only the adapted answer: no preamble, no explanation, no mention of the cache.

Original query: {old_q}
Cached answer:
{answer}

New query: {new_q}
Differences: {diffs}"""


PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT}


def patch(provider: Provider, old_q: str, answer: str, new_q: str, diffs: list, max_ratio: float,
          prompt: str = PROMPT) -> tuple[str | None, Completion | None]:
    """Return (patched text or None if rejected, raw completion for cost accounting)."""
    text, c, _ = patch_detail(provider, old_q, answer, new_q, diffs, max_ratio, prompt)
    return text, c


def patch_detail(provider: Provider, old_q: str, answer: str, new_q: str, diffs: list, max_ratio: float,
                 prompt: str = PROMPT) -> tuple[str | None, Completion, str]:
    """Like patch(), but also says why a rejected patch was rejected: one of cannot_patch, empty, too_long,
    too_short, or "ok". The raw completion is returned so callers can inspect what the model wrote."""
    text = prompt.format(cp=CANNOT_PATCH, old_q=old_q, answer=answer, new_q=new_q,
                         diffs="; ".join(str(d) for d in diffs))
    c = provider.complete([{"role": "user", "content": text}])
    ok, reason = check(c.text, answer, new_q, max_ratio)
    return ok, c, reason


# Markup (table pipes, list dashes, code fences) adds a roughly fixed number of characters, which dominates on a
# short answer: a correct 3-row table of a 113-character answer is 224 characters. A pure ratio rejected such
# patches, so the limit is the larger of the ratio and the cached length plus this slack. Long answers are
# unaffected because the ratio is the larger term.
MARKUP_SLACK = 150

# Requests whose correct answer may legitimately be much shorter than the cached one.
_SHORTENING = ("brief", "short", "concise", "tl;dr", "eli5", "list", "bullet", "table", "json", "yaml", "csv",
               "summar", "one sentence", "one line", "in a sentence")


def check(text: str, cached: str, new_q: str, max_ratio: float) -> tuple[str | None, str]:
    t = text.strip()
    if not t:
        return None, "empty"
    if CANNOT_PATCH in t:
        return None, "cannot_patch"
    if len(t) > max(max_ratio * len(cached), len(cached) + MARKUP_SLACK):
        return None, "too_long"  # grew too much: probably new content
    if len(t) < 0.2 * len(cached) and not any(w in new_q.lower() for w in _SHORTENING):
        return None, "too_short"
    return t, "ok"


def validate(text: str, cached: str, new_q: str, max_ratio: float) -> str | None:
    return check(text, cached, new_q, max_ratio)[0]
