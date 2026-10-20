import numpy as np


def top_k(query_vec: np.ndarray, vecs: list[np.ndarray], k: int) -> list[tuple[int, float]]:
    """Exact cosine top-k over L2-normalized vectors. Returns (position, cosine)."""
    if not vecs:
        return []
    sims = np.stack(vecs) @ query_vec
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]
