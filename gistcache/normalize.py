import re

_BOILERPLATE = re.compile(r"^(hi|hello|hey|please|pls|can you|could you|would you)[,!\s]+", re.I)


def normalize(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    prev = None
    while prev != t:
        prev = t
        t = _BOILERPLATE.sub("", t)
    return t.strip(" ?!.")
