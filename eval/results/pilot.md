# GistCache pilot

96 queries run. API spend $0.3051 of $0.35 cap (large 96 calls $0.2903, small 121 calls $0.0148).

## Tier by query type

| Query type | n | Expected | hit | near_hit | miss | Over-reuse |
|---|---|---|---|---|---|---|
| seed | 24 | miss | 0 | 0 | 24 | 0 |
| paraphrase | 24 | hit | 12 | 3 | 9 | 0 |
| patchable | 24 | near_hit | 0 | 20 | 4 | 0 |
| critical | 24 | miss | 0 | 0 | 24 | 0 |

## Quality of reused answers (LLM judge vs fresh reference)

- hit: 12/12 acceptable (0 not judged)
- near_hit: 22/23 acceptable (0 not judged)
  - NOT acceptable [near_hit, patchable]: 'In a short table, how should I split rent fairly with roommates?' (near_hit: output format) -> 'NO: the query asked for a short table, but the candidate gives two tables plus a numbered list, and its example never computes actual dollar'

## Cost and latency

- GistCache pipeline cost: $0.1953
- No-cache estimate (96 x mean large call $0.00302): $0.2903
- Saved: 33%
- mean latency hit: 115 ms (n=12)
- mean latency miss: 4235 ms (n=61)
- mean latency near_hit: 3388 ms (n=23)
