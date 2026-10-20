"""Stage 1: deterministic difference extraction between two queries."""
import re
from dataclasses import dataclass

CRITICAL = "critical"
PATCHABLE = "patchable"

# --- negation ---------------------------------------------------------------------------------
_NEG = re.compile(
    r"\b(?:not|no(?!\.)|never|without|cannot|nothing|none|neither|nor|non(?=-)"
    r"|\w+n't|wont|dont|doesnt|didnt|isnt|arent|cant|shouldnt|wouldnt|couldnt|wasnt|werent|hasnt|havent)\b")
_NEG_PREFIXES = ("un", "il", "im", "in", "ir", "dis", "non", "de", "anti")

# Opposite pairs. Stems are compared, so "blocking"/"blocks" match "block".
_ANTONYMS = [
    ("allow", "block"), ("allow", "deny"), ("allow", "forbid"), ("enable", "disable"),
    ("increase", "decrease"), ("raise", "lower"), ("win", "lose"), ("lock", "unlock"),
    ("speed", "slow"), ("buy", "sell"), ("show", "hide"), ("add", "remove"), ("open", "close"),
    ("start", "stop"), ("begin", "end"), ("include", "exclude"), ("accept", "reject"),
    ("safe", "dangerous"), ("hot", "cold"), ("push", "pull"), ("encrypt", "decrypt"),
    ("install", "uninstall"), ("true", "false"), ("first", "last"), ("more", "less"),
    ("most", "least"), ("best", "worst"), ("high", "low"), ("fast", "slow"), ("big", "small"),
    ("large", "small"), ("early", "late"), ("good", "bad"), ("pro", "con"), ("positive", "negative"),
    ("male", "female"), ("legal", "illegal"), ("succeed", "fail"), ("pass", "fail"), ("gain", "lose"),
    ("rise", "fall"), ("expand", "shrink"), ("borrow", "lend"), ("win", "lose"), ("agree", "disagree"),
    ("advantage", "disadvantage"), ("benefit", "drawback"), ("buy", "rent"), ("hire", "fire"),
    ("login", "logout"), ("sign in", "sign out"), ("upload", "download"), ("minimum", "maximum"),
]

# --- direction / roles ------------------------------------------------------------------------
_DIRECTION = [("to", "from"), ("into", "out of"), ("before", "after"), ("up", "down"),
              ("import", "export"), ("encode", "decode"), ("compress", "decompress")]

# --- languages, numbers, time -----------------------------------------------------------------
_LANGS = ["python", "java", "javascript", "typescript", "rust", "golang", "c++", "c#",
          "ruby", "php", "swift", "kotlin", "scala", "sql", "bash", "perl", "haskell", "dart",
          "lua", "matlab", "es5", "es6", "html", "css"]
_NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
_NUMBER_WORDS.update({"thirty": 30, "forty": 40, "fifty": 50, "hundred": 100, "thousand": 1000,
                      "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
                      "once": 1, "twice": 2, "thrice": 3, "half": 0.5, "double": 2, "triple": 3})
_NUM = re.compile(r"\d+(?:\.\d+)?")
_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")

_TIME = re.compile(
    r"\b(?:latest|today|current|currently|now|price|prices|recent|news|tonight|tomorrow|yesterday"
    r"|this (?:week|month|year|morning|evening)|right now|at the moment|nowadays|these days|as of"
    r"|(?:the )?weather (?:in|at|for|near|today|tomorrow|tonight|this|like|forecast|right|outside)|how(?:'s| is) the weather|forecast|exchange rate|traffic|trending|breaking|standings|scoreboard|who won"
    r"|stock (?:price|prices|market|quote)|share price|live score|latest score|what time is it"
    r"|who is (?:the )?(?:current )?(?:ceo|president|prime minister|mayor|governor|leader|owner|coach"
    r"|manager|chairman|head)(?: of)?"
    r"|is (?:the |it |this )?\w+ (?:open|closed)(?: now)?"
    r"|open now)\b")

# --- patchable differences --------------------------------------------------------------------
_FORMAT_WORDS = ["table", "json", "yaml", "csv", "markdown", "bullet", "paragraph", "xml", "html"]
# "list" is also a noun/verb ("sort a list"), so it only counts as a format when requested as one.
_LIST_FORMAT = re.compile(r"\b(?:as|in|into)\s+(?:a\s+|an\s+)?(?:bulleted\s+|numbered\s+)?list\b")
_FORMATS = _FORMAT_WORDS + ["list"]

_ORDER = re.compile(
    r"\b(ascending|descending|asc|desc|increasing|decreasing|newest first|oldest first|latest first"
    r"|most recent first|largest first|smallest first|highest first|lowest first|biggest first"
    r"|reverse order|reversed|reverse|alphabetical(?:ly)?|a-z|z-a)\b")

_VERBOSITY = re.compile(
    r"\b(briefly|brief|short(?: version| answer)?|concise(?:ly)?|detailed|in detail|eli5|like i'?m five"
    r"|step by step|verbose|tl;?dr|simply|simple terms?|in simple|one[- ]line|one sentence|in a sentence"
    r"|one paragraph|summary|for beginners?|for a beginner|for experts?|advanced|casual(?:ly)?"
    r"|formal(?:ly)?|professional(?:ly)?|thorough(?:ly)?|in depth|quick(?:ly)? overview)\b")

_CONSTRAINT = re.compile(
    r"\b(?:with|including|includes?|add|adding|using|plus)\s+(?:an?\s+|some\s+|the\s+)?(?:\w+\s+)?"
    r"(?:examples?|comments?|type hints?|docstrings?|tests?|error handling|explanations?|citations?"
    r"|sources?|links?|references?|proofs?|diagrams?|code|steps|pros and cons)\b")

# Function words and conversational filler. Anything else is a "content" word.
_FILLER = set("""a an the is are was were be been am do does did how what which who whom whose why when where
can could would should will shall may might must i me my we our you your it its this that these those
to of in on at by for with from as and or but if so than then there here please pls hi hello hey
tell give show explain describe know want need help tips tip ways way any some about into onto up
out really just also very much many more get make let lets like would wonder wondering curious idea
ideas thoughts question questions answer anyone someone thing things kind sort quickly basically between versus vs""".split())


@dataclass(frozen=True)
class Difference:
    kind: str
    severity: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}" if self.detail else self.kind


def _toks(t: str) -> list[str]:
    return [w[:-2] if w.endswith("'s") and len(w) > 3 else w for w in re.findall(r"[a-z0-9+#']+", t.lower())]


def _words(t: str) -> set[str]:
    return set(_toks(t))


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def _is_filler(w: str) -> bool:
    return (w in _FILLER or w[:-1] in _FILLER or w[:-2] in _FILLER or w + "e" in _FILLER
            or (w.endswith("ing") and (w[:-3] in _FILLER or w[:-3] + "e" in _FILLER)))


_PATCHABLE_PHRASES = (_ORDER, _VERBOSITY, _CONSTRAINT)


def _core(t: str) -> str:
    """Text with patchable phrases removed, so they do not count as numbers or extra content."""
    low = t.lower()
    for rx in _PATCHABLE_PHRASES:
        low = rx.sub(" ", low)
    low = _LIST_FORMAT.sub(" ", low)
    return re.sub(r"\s+", " ", low)


def _neg_set(t: str) -> set[str]:
    # Presence only: "no X", "not X", "non-X" and "without X" all negate; which word is used is style.
    return {"neg"} if _NEG.search(t.lower().replace("’", "'")) else set()


def _prefix_negation(wa: set[str], wb: set[str]) -> str | None:
    only_a, only_b = wa - wb, wb - wa
    for x in only_a:
        for y in only_b:
            for s, l in ((x, y), (y, x)):
                for p in _NEG_PREFIXES:
                    if l == p + s and len(s) >= 4:
                        return f"{s}/{l}"
    return None


def _antonym(wa: set[str], wb: set[str]) -> str | None:
    sa, sb = {_stem(w) for w in wa}, {_stem(w) for w in wb}
    for x, y in _ANTONYMS:
        x, y = _stem(x), _stem(y)
        if " " in x or " " in y:
            continue
        if (x in sa and y in sb and x not in sb and y not in sa) or \
           (y in sa and x in sb and y not in sb and x not in sa):
            return f"{x}/{y}"
    return None


# "go", "c" and "r" are ordinary words too, so they only count in a language context.
_AMBIGUOUS_LANG = re.compile(r"\b(?:in|using|with|for|on)\s+(go|c|r)\b|\b(go|c|r)\s+(?:language|lang|programming|code)\b")


def _langs(t: str) -> set[str]:
    low = t.lower()
    found = {l for l in _LANGS if l in _words(t)}
    for m in _AMBIGUOUS_LANG.finditer(low):
        found.add({"go": "golang"}.get(m.group(1) or m.group(2), m.group(1) or m.group(2)))
    return found


def _caps(t: str) -> set[str]:
    # Capitalized tokens that do not start a sentence, as a cheap entity proxy.
    out, start = set(), True
    for x in re.findall(r"\S+", t):
        w = x.strip(".,?!:;\"'()").lower()
        if not start and x[:1].isupper() and len(w) > 1 and "'" not in x:
            out.add(w)
        start = x[-1:] in ".?!:"
    return out - set(_FORMATS)


def _numbers(t: str) -> list[float]:
    # "one" is excluded: it is mostly a pronoun ("which one", "no one").
    core = _core(t)
    out = [float(x) for x in _NUM.findall(core)]
    out += [float(_NUMBER_WORDS[w]) for w in _toks(core) if w in _NUMBER_WORDS and w != "one"]
    out += [-1.0 for w in _toks(core) if w in ("minus", "negative")]
    return sorted(out)


def _formats(lower: str, words: set[str]) -> set[str]:
    found = {f for f in _FORMAT_WORDS if f in words}
    if "bullets" in words:
        found.add("bullet")
    if _LIST_FORMAT.search(lower):
        found.add("list")
    return found


def _content(t: str) -> set[str]:
    return {_stem(w) for w in _toks(_core(t)) if not _is_filler(w) and not w.isdigit() and len(w) > 1}


def extract_differences(a: str, b: str, strict_content: bool = True) -> list[Difference]:
    """Differences between cached query `a` and new query `b` (original text).

    `strict_content` enables the safety net that refuses a plain hit when one query adds or drops
    content words no rule explains. It costs some hits (they become near-hits) and buys safety."""
    la, lb = a.lower(), b.lower()
    wa, wb = _words(a), _words(b)
    diffs: list[Difference] = []

    if _TIME.search(la) or _TIME.search(lb):
        diffs.append(Difference("time_sensitive", CRITICAL, "time-sensitive terms"))

    if _neg_set(a) != _neg_set(b):
        diffs.append(Difference("negation", CRITICAL, "negation mismatch"))
    elif (pn := _prefix_negation(wa, wb)):
        diffs.append(Difference("negation", CRITICAL, f"prefix negation {pn}"))
    elif (an := _antonym(wa, wb)):
        diffs.append(Difference("negation", CRITICAL, f"antonym {an}"))

    cwa, cwb = _content(a), _content(b)
    overlap = len(cwa & cwb) / max(len(cwa | cwb), 1)
    for x, y in _DIRECTION:
        if ((x in wa) != (x in wb) or (y in wa) != (y in wb)) and ((x in wa or x in wb) and (y in wa or y in wb)) \
                and overlap >= 0.5:
            diffs.append(Difference("direction", CRITICAL, f"{x}/{y}"))
            break
    # Same words in a different order. Only critical when the words are not symmetric ("A and B").
    if not any(d.kind == "direction" for d in diffs) and wa == wb and la != lb:
        ta, tb = _toks(la), _toks(lb)
        if ta != tb and not (set(ta) & {"and", "or", "versus", "vs", "between"}):
            diffs.append(Difference("direction", CRITICAL, "word order swapped"))

    if _numbers(a) != _numbers(b) or sorted(_ORDINAL.findall(a)) != sorted(_ORDINAL.findall(b)):
        diffs.append(Difference("number", CRITICAL, f"{_numbers(a)} -> {_numbers(b)}"))

    if _langs(a) != _langs(b):
        diffs.append(Difference("language", CRITICAL, f"{sorted(_langs(a))} -> {sorted(_langs(b))}"))

    ea, eb = _caps(a) - wb, _caps(b) - wa  # "String" vs "string" is the same word, not a new entity
    if ea != eb and (ea | eb) - (ea & eb) - _langs(a) - _langs(b):
        diffs.append(Difference("entity", CRITICAL, f"{sorted(ea - eb)} -> {sorted(eb - ea)}"))

    fa, fb = _formats(la, wa), _formats(lb, wb)
    if fa != fb:
        diffs.append(Difference("output_format", PATCHABLE, f"{sorted(fa)} -> {sorted(fb)}"))

    oa, ob = set(_ORDER.findall(la)), set(_ORDER.findall(lb))
    if oa != ob:
        diffs.append(Difference("sort_order", PATCHABLE, f"{sorted(oa)} -> {sorted(ob)}"))

    va, vb = set(_VERBOSITY.findall(la)), set(_VERBOSITY.findall(lb))
    if va != vb:
        diffs.append(Difference("verbosity", PATCHABLE, f"{sorted(va)} -> {sorted(vb)}"))

    na = {m.group(0) for m in _CONSTRAINT.finditer(la)}
    nb = {m.group(0) for m in _CONSTRAINT.finditer(lb)}
    if na != nb:
        diffs.append(Difference("constraint", PATCHABLE, f"{sorted(nb - na)}"))

    # Safety net: if one query's content words are a strict subset of the other's, the extra words
    # are content the cached answer was not written for. Never serve that as a plain hit.
    ca, cb = _content(a), _content(b)
    # A near-hit is only explained by its patchable phrase if everything else is the same. If each side also
    # has content words the other lacks ("brave" vs "cowardly"), a swap is hiding behind the patchable
    # difference, so mark it for the contradiction guard instead of trusting the explanation.
    if (ca - cb) and (cb - ca) and any(d.severity == PATCHABLE for d in diffs):
        diffs.append(Difference("content_substituted", PATCHABLE, f"{sorted(ca - cb)} -> {sorted(cb - ca)}"))
    if strict_content and ca and cb and ca != cb:
        if ca < cb:
            diffs.append(Difference("content_added", PATCHABLE, "+" + ",".join(sorted(cb - ca))))
        elif cb < ca:
            diffs.append(Difference("content_removed", PATCHABLE, "-" + ",".join(sorted(ca - cb))))

    return diffs


def is_time_sensitive(text: str) -> bool:
    return bool(_TIME.search(text.lower()))
