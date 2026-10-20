"""Decide which real user prompts are usable as evaluation seeds.

A seed must be a short, self-contained, single-turn English question or request with a stable answer, and must
not contain personal data. Every rejection has a reason so the filter can be audited.
"""
import re

import numpy as np

from gistcache.normalize import normalize
from gistcache.verifier.rules import is_time_sensitive

MIN_CHARS, MAX_CHARS, MIN_WORDS = 20, 300, 4

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?:\+?\d[\s().-]?){9,}")
_URL = re.compile(r"https?://|www\.", re.I)
_SECRET = re.compile(r"\b(?:sk-[A-Za-z0-9]{10,}|AKIA[0-9A-Z]{12,}|ghp_[A-Za-z0-9]{20,}|[A-Fa-f0-9]{32,})\b")
_CODE = re.compile(r"```|[{};]\s*$|^\s*(def|class|import|function|public|#include)\b", re.M)
# Role-play, jailbreaks and "continue the text" tasks have no single correct answer to reuse.
_NOT_A_QUESTION = re.compile(
    r"\b(?:pretend|roleplay|role-play|act as|you are (?:a|an|now)|ignore (?:all|previous)|jailbreak|DAN\b"
    r"|continue the (?:story|text)|nsfw"
    r"|write (?:me )?(?:a|an|some) (?:\w+ ){0,2}(?:story|stories|poem|essay|song|novel|joke|article|ad|script|limerick))\b",
    re.I)
# Machine-generated prompt templates that recur thousands of times in public chat logs (red-teaming probes,
# SEO article generators). They are not real user questions, and repeated templates would make seeds hit each
# other, inflating any cache's hit rate.
_TEMPLATE = re.compile(
    r"\[your answer\]|say something (?:toxic|bad|harmful|benign|positive|negative)\b|\(no more than \d+ words\)"
    r"|here is my query\s*:|\b(?:negative|harmful|toxic) (?:or (?:negative|harmful|toxic) )?comment"
    r"|\bin (?:the )?chemical industry\b|introduction (?:of )?over \d+ words for|\b\d{3,4} words\b", re.I)


def reject_reason(text: str) -> str | None:
    """None if the prompt is usable as a seed, otherwise a short reason."""
    t = text.strip()
    if not MIN_CHARS <= len(t) <= MAX_CHARS:
        return "length"
    if len(t.split()) < MIN_WORDS:
        return "too_few_words"
    if _EMAIL.search(t) or _PHONE.search(t) or _SECRET.search(t):
        return "personal_data"  # checked first: privacy matters more than any other reason
    if sum(c.isascii() for c in t) / len(t) < 0.97:
        return "non_ascii"  # cheap backstop; the dataset's language tag does the real English filtering
    if sum(c.isalpha() or c.isspace() for c in t) / len(t) < 0.8:
        return "symbols"
    if _URL.search(t):
        return "url"
    if _CODE.search(t) or "\n" in t:
        return "code_or_multiline"
    if _TEMPLATE.search(t):
        return "template"
    if t.endswith(":") or re.search(r"\bwait for my (?:next )?(?:prompt|message|input)\b", t, re.I):
        return "needs_context"  # refers to text the user was about to paste, or sets up a later turn
    if _NOT_A_QUESTION.search(t):
        return "not_a_question"
    if is_time_sensitive(t):
        return "time_sensitive"
    return None


def dedupe(prompts: list[str], embedder, threshold: float = 0.9) -> list[str]:
    """Drop prompts that are exact or near duplicates (cosine >= threshold) of one already kept."""
    kept, vecs, seen = [], [], set()
    for p in prompts:
        key = normalize(p)
        if key in seen:
            continue
        v = embedder.encode(key)
        if vecs and float(np.max(np.stack(vecs) @ v)) >= threshold:
            continue
        seen.add(key)
        kept.append(p)
        vecs.append(v)
    return kept
