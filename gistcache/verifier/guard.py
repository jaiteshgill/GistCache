"""Contradiction guard: a veto on would-be hits.

An NLI model scores whether the two queries contradict each other. Lexicon rules cannot list every
antonym or same-slot entity swap ("attract/repel", "iphone/android phone"); a contradiction model
generalizes. It only runs on candidate hits, so it adds no cost to near-hits and misses.
"""
from typing import Protocol


class Guard(Protocol):
    def contradiction(self, a: str, b: str) -> float: ...


class NLIGuard:
    # cross-encoder NLI label order: contradiction, entailment, neutral
    def __init__(self, model: str = "cross-encoder/nli-deberta-v3-small"):
        from sentence_transformers import CrossEncoder  # optional dependency
        self._m = CrossEncoder(model)

    def contradiction(self, a: str, b: str) -> float:
        p = self._m.predict([(a, b), (b, a)], apply_softmax=True)
        return float(max(p[0][0], p[1][0]))
