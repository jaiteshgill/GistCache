"""The comparison every semantic cache is measured against: one cosine threshold, nothing else."""
import threading
import time

import numpy as np

from gistcache.normalize import normalize


class Memo:
    """Answers each distinct large-model request once and replays it afterwards.

    GistCache, the threshold baseline and the judge's reference answer then all see the same answer for the same
    question, which makes the comparison fair and avoids paying for one question three times. The replay keeps
    the original latency so a system that hits the memo is not reported as faster than it would really be."""

    def __init__(self, inner):
        self.inner, self.model = inner, inner.model
        self._seen, self._lock = {}, threading.Lock()
        self.last_latency_ms = 0.0

    def complete(self, messages, **params):
        key = repr((messages, sorted(params.items())))
        with self._lock:
            hit = self._seen.get(key)
        if hit is None:
            t0 = time.perf_counter()
            c = self.inner.complete(messages, **params)
            hit = (c, (time.perf_counter() - t0) * 1000)
            with self._lock:
                self._seen.setdefault(key, hit)
        self.last_latency_ms = hit[1]
        return hit[0]

    def latencies(self) -> list[float]:
        return [ms for _, ms in self._seen.values()]


class ThresholdCache:
    """Serve the cached answer of the most similar earlier query if cosine >= threshold, else ask the large model."""

    def __init__(self, embedder, large: Memo, threshold: float, pricing: dict):
        self.embedder, self.large, self.threshold, self.pricing = embedder, large, threshold, pricing
        self.vecs, self.answers = [], []

    def handle(self, query: str, max_tokens: int) -> dict:
        from gistcache.cost import cost_usd
        t0 = time.perf_counter()
        v = self.embedder.encode(normalize(query))
        if self.vecs:
            sims = np.stack(self.vecs) @ v
            i = int(np.argmax(sims))
            if sims[i] >= self.threshold:
                return {"tier": "hit", "text": self.answers[i], "cost": 0.0, "cos": float(sims[i]),
                        "ms": (time.perf_counter() - t0) * 1000}
        c = self.large.complete([{"role": "user", "content": query}], max_tokens=max_tokens)
        self.vecs.append(v)
        self.answers.append(c.text)
        own = (time.perf_counter() - t0) * 1000
        return {"tier": "miss", "text": c.text, "cost": cost_usd(self.pricing, "large", c.in_tokens, c.out_tokens),
                "ms": own + self.large.last_latency_ms}
