"""Stage 1: deterministic difference extraction between two queries."""
import re
from dataclasses import dataclass

CRITICAL = "critical"
PATCHABLE = "patchable"

_NEG = re.compile(r"\b(not|no|never|without|don'?t|doesn'?t|didn'?t|isn'?t|aren'?t|won'?t|can'?t|cannot|shouldn'?t)\b")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_DIRECTION = [("to", "from"), ("into", "out of"), ("before", "after"), ("up", "down"),
              ("import", "export"), ("encode", "decode"), ("compress", "decompress")]
_LANGS = ["python", "java", "javascript", "typescript", "rust", "go", "golang", "c++", "c#",
          "ruby", "php", "swift", "kotlin", "scala", "sql", "bash", "perl", "haskell"]
_VERSION = re.compile(r"\b(?:v|version\s*)?(\d+(?:\.\d+)*)\b")
_TIME = re.compile(r"\b(latest|today|current|currently|now|price|prices|recent|news|tonight|tomorrow|yesterday|this (?:week|month|year))\b")

_FORMATS = ["list", "table", "json", "yaml", "csv", "markdown", "bullet", "paragraph", "xml"]
_ORDERS = [("ascending", "descending"), ("asc", "desc"), ("increasing", "decreasing")]
_VERBOSITY = re.compile(r"\b(briefly|brief|short|concise|detailed|in detail|eli5|like i'?m five|step by step|verbose|tl;?dr)\b")
_NARROW = re.compile(r"\b(with|add|using|include)\s+(comments|type hints|docstrings|tests|examples|error handling)\b")


@dataclass(frozen=True)
class Difference:
    kind: str
    severity: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}" if self.detail else self.kind


def _words(t: str) -> set[str]:
    return set(re.findall(r"[a-z0-9+#']+", t.lower()))


def _neg_set(t: str) -> set[str]:
    return {m.group(0).replace("'", "") for m in _NEG.finditer(t.lower())}


def _langs(t: str) -> set[str]:
    w = _words(t)
    return {l for l in _LANGS if l in w}


def _caps(t: str) -> set[str]:
    # Capitalized tokens not at sentence start, as a cheap entity proxy.
    toks = re.findall(r"\S+", t)
    return {x.strip(".,?!\"'").lower() for i, x in enumerate(toks) if i > 0 and x[:1].isupper() and len(x) > 1}


def _order_of(t: str, a: str, b: str) -> str | None:
    w = _words(t)
    return a if a in w else b if b in w else None


def extract_differences(a: str, b: str) -> list[Difference]:
    """Differences between cached query `a` and new query `b` (original text)."""
    la, lb = a.lower(), b.lower()
    wa, wb = _words(a), _words(b)
    diffs: list[Difference] = []

    if _TIME.search(la) or _TIME.search(lb):
        diffs.append(Difference("time_sensitive", CRITICAL, "time-sensitive terms"))

    if _neg_set(a) != _neg_set(b):
        diffs.append(Difference("negation", CRITICAL, "negation mismatch"))

    for x, y in _DIRECTION:
        if ((x in wa) != (x in wb) or (y in wa) != (y in wb)) and ((x in wa or x in wb) and (y in wa or y in wb)):
            diffs.append(Difference("direction", CRITICAL, f"{x}/{y}"))
            break
    # Same swapped words in a different order, e.g. "CAD to USD" vs "USD to CAD".
    if not any(d.kind == "direction" for d in diffs) and wa == wb and la != lb:
        ta, tb = re.findall(r"[a-z0-9+#']+", la), re.findall(r"[a-z0-9+#']+", lb)
        if ta != tb:
            diffs.append(Difference("direction", CRITICAL, "word order swapped"))

    if _NUM.findall(a) != _NUM.findall(b):
        diffs.append(Difference("number", CRITICAL, f"{_NUM.findall(a)} -> {_NUM.findall(b)}"))

    if _langs(a) != _langs(b):
        diffs.append(Difference("language", CRITICAL, f"{sorted(_langs(a))} -> {sorted(_langs(b))}"))

    ea, eb = _caps(a), _caps(b)
    if ea != eb and (ea | eb) - (ea & eb) - _langs(a) - _langs(b):
        diffs.append(Difference("entity", CRITICAL, f"{sorted(ea - eb)} -> {sorted(eb - ea)}"))

    fa, fb = {f for f in _FORMATS if f in wa}, {f for f in _FORMATS if f in wb}
    if fa != fb:
        diffs.append(Difference("output_format", PATCHABLE, f"{sorted(fa)} -> {sorted(fb)}"))

    for x, y in _ORDERS:
        oa, ob = _order_of(a, x, y), _order_of(b, x, y)
        if oa != ob:
            diffs.append(Difference("sort_order", PATCHABLE, f"{oa} -> {ob}"))
            break

    va, vb = set(_VERBOSITY.findall(la)), set(_VERBOSITY.findall(lb))
    if va != vb:
        diffs.append(Difference("verbosity", PATCHABLE, f"{sorted(va)} -> {sorted(vb)}"))

    na = {m.group(0) for m in _NARROW.finditer(la)}
    nb = {m.group(0) for m in _NARROW.finditer(lb)}
    if na != nb:
        diffs.append(Difference("constraint", PATCHABLE, f"{sorted(nb - na)}"))

    return diffs


def is_time_sensitive(text: str) -> bool:
    return bool(_TIME.search(text.lower()))
