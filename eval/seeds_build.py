"""Build evaluation seeds from a real-prompt dataset.

  hf auth login                                                  # once, with a read token (or export HF_TOKEN)
  python -m eval.seeds_build --source lmsys --inspect            # print the schema of one row (no content)
  python -m eval.seeds_build --source lmsys --n 150              # -> eval/data/seeds_lmsys.json

LMSYS-Chat-1M is gated and its license forbids redistribution, so the output goes to eval/data/ (gitignored).
Never commit it, and do not quote its prompts in anything published. `--source dolly` uses the openly licensed
databricks-dolly-15k instead.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .seed_filter import dedupe, reject_reason

HERE = Path(__file__).parent
MODERATION_MAX = 0.1


def lmsys_prompt(row: dict) -> tuple[str | None, str | None]:
    """(prompt, None) for a usable single-turn English row, else (None, reason). Field access is defensive
    because the dataset card does not document the exact schema."""
    msgs = row.get("conversation") or []
    users = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
    if len(users) != 1 or row.get("turn", 1) != 1:
        return None, "multi_turn"
    if str(row.get("language", "")).lower() != "english":
        return None, "not_english"
    if row.get("redacted"):
        return None, "redacted"
    mods = [m for m in (row.get("openai_moderation") or []) if isinstance(m, dict)]
    if any(m.get("flagged") for m in mods):
        return None, "moderation_flagged"
    # "flagged" is a high bar; borderline sexual, violent or hateful prompts pass it. Any category score at or
    # above MODERATION_MAX rejects the row too.
    if any(float(v) >= MODERATION_MAX for m in mods for v in (m.get("category_scores") or {}).values()):
        return None, "moderation_score"
    return str(users[0].get("content", "")), None


def dolly_prompt(row: dict) -> tuple[str | None, str | None]:
    if (row.get("context") or "").strip():
        return None, "needs_context"  # the answer depends on a passage we would not have
    if row.get("category") not in ("open_qa", "general_qa", "brainstorming", "classification"):
        return None, "category"
    return str(row.get("instruction", "")), None


SOURCES = {
    "lmsys": ("lmsys/lmsys-chat-1m", lmsys_prompt),
    "dolly": ("databricks/databricks-dolly-15k", dolly_prompt),
}


def select(rows, extract, n: int, embedder, rng: random.Random, max_scan: int, pool_factor: int = 4):
    """Scan rows, keep usable prompts until there is a pool of n * pool_factor, dedupe, then sample n."""
    reasons, pool, scanned = Counter(), [], 0
    for row in rows:
        scanned += 1
        text, why = extract(row)
        if why is None:
            why = reject_reason(text)
        if why:
            reasons[why] += 1
        else:
            pool.append(text.strip())
        if len(pool) >= n * pool_factor or scanned >= max_scan:
            break
    unique = dedupe(pool, embedder)
    reasons["near_duplicate"] += len(pool) - len(unique)
    seeds = rng.sample(unique, min(n, len(unique)))
    return seeds, {"scanned": scanned, "pool": len(pool), "unique": len(unique), "rejected": dict(reasons)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=sorted(SOURCES), default="lmsys")
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--max-scan", type=int, default=60000)
    p.add_argument("--rng", type=int, default=7)
    p.add_argument("--inspect", action="store_true", help="print the keys and types of the first row, no content")
    a = p.parse_args()
    from datasets import load_dataset  # optional dependency, imported only when used

    name, extract = SOURCES[a.source]
    rows = load_dataset(name, split="train", streaming=True)
    if a.inspect:
        row = next(iter(rows))
        for k, v in row.items():
            inner = f" of {type(v[0]).__name__} with keys {sorted(v[0])}" if isinstance(v, list) and v and isinstance(v[0], dict) else ""
            print(f"  {k}: {type(v).__name__}{inner}")
        return

    from gistcache.embed import SentenceTransformerEmbedder
    seeds, stats = select(rows, extract, a.n, SentenceTransformerEmbedder(), random.Random(a.rng), a.max_scan)
    out = HERE / "data" / f"seeds_{a.source}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"source": name, **stats, "seeds": seeds}, indent=1))
    print(f"scanned {stats['scanned']} rows, {stats['pool']} usable, {stats['unique']} after dedupe -> {len(seeds)} seeds")
    print("rejected:", ", ".join(f"{k} {v}" for k, v in Counter(stats["rejected"]).most_common()))
    print(f"wrote {out} (gitignored; do not commit or publish its prompts)")


if __name__ == "__main__":
    main()
