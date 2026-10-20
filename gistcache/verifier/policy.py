"""Stage 3: tier decision."""
from dataclasses import dataclass, field

from .rules import CRITICAL, Difference

HIT, NEAR_HIT, MISS = "hit", "near_hit", "miss"


@dataclass
class Decision:
    tier: str
    reason: str
    score: float = 0.0
    differences: list[Difference] = field(default_factory=list)


def decide(diffs: list[Difference], score: float, hit_threshold: float, near_threshold: float,
           cosine: float | None = None) -> Decision:
    """`score` is the cross-encoder (duplicate-detection) score. It gates hits only.
    Near-hits are gated on cosine: the rules already explained the difference, and a duplicate
    detector would (correctly) call "sort ascending" and "sort descending" non-duplicates."""
    gate = score if cosine is None else cosine
    crit = [d for d in diffs if d.severity == CRITICAL]
    if crit:
        return Decision(MISS, f"miss: {crit[0].detail or crit[0].kind}" if crit[0].kind != "entity"
                        else f"miss: entity {crit[0].detail}", score, diffs)
    if not diffs and score >= hit_threshold:
        return Decision(HIT, "hit: paraphrase", score, diffs)
    if diffs and gate >= near_threshold:
        return Decision(NEAR_HIT, f"near_hit: {diffs[0].kind.replace('_', ' ')}", score, diffs)
    if not diffs and score >= near_threshold:
        return Decision(MISS, "miss: below hit threshold", score, diffs)
    return Decision(MISS, "miss: low similarity", score, diffs)


def needs_guard(d: Decision) -> bool:
    """Hits always get the contradiction check. So do near-hits no rule explains: when the only
    differences are the generic content_added/content_removed net, the change could be an antonym."""
    if d.tier == HIT:
        return True
    return d.tier == NEAR_HIT and bool(d.differences) and all(x.kind.startswith("content_") for x in d.differences)
