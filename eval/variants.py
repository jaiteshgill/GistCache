"""Generate labelled variants of each seed with the small model. Cached on disk so reruns are free."""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROMPT = """You are building a test set for a semantic cache. For the user prompt below, write three variants.

1. "paraphrase": exactly the same meaning in different words. Do not add or remove any requirement.
2. "patchable": the same question, but ONLY the output format, sort order, verbosity, or one narrowing
   constraint changes (e.g. "as a table", "briefly", "with comments", "in descending order").
3. "critical": a change that makes a cached answer WRONG: negate it, swap the entity, change a number,
   version or programming language, or reverse a direction.

Reply with only a JSON object with keys "paraphrase", "patchable", "critical". No commentary.

Prompt: {seed}"""


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return d if all(isinstance(d.get(k), str) and d[k].strip() for k in ("paraphrase", "patchable", "critical")) else None


def _one(small, seed: str) -> dict | None:
    for _ in range(2):  # one retry on malformed output
        d = parse(small.complete([{"role": "user", "content": PROMPT.format(seed=seed)}], max_tokens=400).text)
        if d:
            return d
    return None


def generate(small, seeds: list[str], cache_path: Path, workers: int = 8) -> dict[str, dict]:
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [s for s in seeds if s not in cache]
    with ThreadPoolExecutor(workers) as ex:
        for s, d in zip(todo, ex.map(lambda s: _one(small, s), todo)):
            if d:
                cache[s] = d
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=1))
    return {s: cache[s] for s in seeds if s in cache}
