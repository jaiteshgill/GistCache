import hashlib
import re

import numpy as np


class HashEmbedder:
    """Dependency-free bag-of-words hashing embedder. For tests and offline dev only."""
    dim = 256

    def encode(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9+#']+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer(model, device=device)

    def encode(self, text: str) -> np.ndarray:
        return self._m.encode(text, normalize_embeddings=True).astype(np.float32)
