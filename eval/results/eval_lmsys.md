# GistCache evaluation (lmsys seeds)

596 queries run. API spend $1.8312 of $2.5 cap (large 596 calls $1.7069, small 932 calls $0.1243).

## End to end (same query stream for every system)

| System | Large model avoided | Reused answers judged acceptable | Wrongly reused | Cost / 1k requests | p50 latency | p95 latency |
|---|---|---|---|---|---|---|
| No cache | 0% | n/a | 0 | $2.86 | 3.19 s | 4.48 s |
| Threshold cache (cosine >= 0.8) | 61% | 131/360 | 229 (38.4 per 100) | $1.13 | 0.01 s | 4.08 s |
| Threshold cache (cosine >= 0.9) | 41% | 87/242 | 155 (26.0 per 100) | $1.70 | 2.69 s | 5.87 s |
| **GistCache** | 22% | 117/128 | 11 (1.8 per 100) | $2.34 | 3.76 s | 6.55 s |

No-cache cost and latency are the mean and percentiles of real large-model calls in this run. "Wrongly reused" counts reused answers the judge rejected.

## GistCache tier by query type

| Query type | n | Expected | hit | near_hit | miss | Over-reuse |
|---|---|---|---|---|---|---|
| seed | 149 | miss | 0 | 0 | 149 | 0 |
| paraphrase | 149 | hit | 51 | 16 | 82 | 0 |
| patchable | 149 | near_hit | 0 | 62 | 87 | 0 |
| critical | 149 | miss | 1 | 0 | 148 | 1 |

## Threshold cache hits by query type

| Threshold | paraphrase hits | patchable hits | critical hits (should be 0) |
|---|---|---|---|
| 0.8 | 123/149 | 139/149 | 103/149 |
| 0.9 | 68/149 | 106/149 | 73/149 |

## GistCache: quality of reused answers (LLM judge vs a large-model reference)

- hit: 46/51 acceptable (1 not judged)
- near_hit: 71/77 acceptable (1 not judged)
  - NOT acceptable [near_hit, patchable] (near_hit: sort order)
  - NOT acceptable [hit, paraphrase] (hit: paraphrase)
  - NOT acceptable [near_hit, patchable] (near_hit: output format)
  - NOT acceptable [hit, paraphrase] (hit: confirmed by judge)
  - NOT acceptable [hit, paraphrase] (hit: paraphrase)
  - NOT acceptable [hit, paraphrase] (hit: paraphrase)
  - NOT acceptable [near_hit, patchable] (near_hit: verbosity)
  - NOT acceptable [hit, paraphrase] (hit: paraphrase)
  - NOT acceptable [near_hit, patchable] (near_hit: output format)
  - NOT acceptable [near_hit, patchable] (near_hit: verbosity)
  - NOT acceptable [near_hit, paraphrase] (near_hit: verbosity)
  - 2 judge replies had no YES/NO and were excluded

## GistCache cost and latency

- GistCache pipeline cost: $1.3958
- No-cache estimate (596 x mean large call $0.00286): $1.7069
- Saved: 18%
- mean latency hit: 513 ms (n=52)
- mean latency miss: 4310 ms (n=466)
- mean latency near_hit: 2538 ms (n=78)

Prompt text is omitted because the source dataset may not be redistributed.

1 of 150 seeds got no usable variants and were excluded.
