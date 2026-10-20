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


def decide(diffs: list[Difference], score: float, hit_threshold: float, near_threshold: float) -> Decision:
    crit = [d for d in diffs if d.severity == CRITICAL]
    if crit:
        return Decision(MISS, f"miss: {crit[0].detail or crit[0].kind}" if crit[0].kind != "entity"
                        else f"miss: entity {crit[0].detail}", score, diffs)
    if not diffs and score >= hit_threshold:
        return Decision(HIT, "hit: paraphrase", score, diffs)
    if diffs and score >= near_threshold:
        return Decision(NEAR_HIT, f"near_hit: {diffs[0].kind.replace('_', ' ')}", score, diffs)
    if not diffs and score >= near_threshold:
        return Decision(MISS, "miss: below hit threshold", score, diffs)
    return Decision(MISS, "miss: low similarity", score, diffs)
