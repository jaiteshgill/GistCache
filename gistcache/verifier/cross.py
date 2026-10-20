"""Stage 2: semantic scoring. Uses a cross-encoder when available, else cosine."""
from typing import Protocol


class Scorer(Protocol):
    def score(self, a: str, b: str, cosine: float) -> float: ...


class CosineScorer:
    def score(self, a: str, b: str, cosine: float) -> float:
        return cosine


class CrossEncoderScorer:
    def __init__(self, model: str = "cross-encoder/quora-distilroberta-base"):
        from sentence_transformers import CrossEncoder  # optional dependency
        self._m = CrossEncoder(model)

    def score(self, a: str, b: str, cosine: float) -> float:
        return float(self._m.predict([(a, b)])[0])
