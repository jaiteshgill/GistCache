"""Spend tracking with a hard cap. Wraps providers so every real API call is counted."""
import threading
from dataclasses import dataclass, field

from gistcache.cost import cost_usd


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Tracker:
    cap_usd: float
    pricing: dict
    spent: dict = field(default_factory=lambda: {"large": 0.0, "small": 0.0})
    calls: dict = field(default_factory=lambda: {"large": 0, "small": 0})
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def total(self) -> float:
        return sum(self.spent.values())

    def wrap(self, provider, role: str):
        return Tracked(provider, role, self)


class Tracked:
    def __init__(self, inner, role: str, tracker: Tracker):
        self.inner, self.role, self.tracker = inner, role, tracker
        self.model = inner.model

    def complete(self, messages, **params):
        t = self.tracker
        if t.total >= t.cap_usd:
            raise BudgetExceeded(f"spend cap ${t.cap_usd:.2f} reached (spent ${t.total:.4f})")
        c = self.inner.complete(messages, **params)
        with t.lock:
            t.spent[self.role] += cost_usd(t.pricing, self.role, c.in_tokens, c.out_tokens)
            t.calls[self.role] += 1
        return c
