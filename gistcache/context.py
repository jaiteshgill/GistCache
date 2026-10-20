import hashlib

from .normalize import normalize

_REFS = ("it", "that", "this", "those", "them", "the second one", "the first one", "the last one", "same")


def split_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Return (latest user query, prior turns)."""
    idx = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=None)
    if idx is None:
        return "", []
    return str(messages[idx].get("content", "")), messages[:idx]


def context_key(prior: list[dict], turns: int = 4) -> str | None:
    if not prior:
        return None
    recent = prior[-turns:]
    canon = "|".join(f"{m.get('role')}:{normalize(str(m.get('content', '')))}" for m in recent)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def has_unresolved_reference(query: str) -> bool:
    words = set(normalize(query).split())
    return any((r in words) if " " not in r else r in normalize(query) for r in _REFS)
