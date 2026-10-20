# GistCache pilot

96 queries run. API spend $0.3037 of $0.35 cap (large 96 calls $0.2891, small 126 calls $0.0146).

## Tier by query type

| Query type | n | Expected | hit | near_hit | miss | Over-reuse |
|---|---|---|---|---|---|---|
| seed | 24 | miss | 0 | 0 | 24 | 0 |
| paraphrase | 24 | hit | 17 | 0 | 7 | 0 |
| patchable | 24 | near_hit | 0 | 18 | 6 | 0 |
| critical | 24 | miss | 0 | 0 | 24 | 0 |

## Quality of reused answers (LLM judge vs fresh reference)

- hit: 17/17 acceptable (0 not judged)
- near_hit: 18/18 acceptable (0 not judged)

## Cost and latency

- GistCache pipeline cost: $0.1954
- No-cache estimate (96 x mean large call $0.00301): $0.2891
- Saved: 32%
- mean latency hit: 423 ms (n=17)
- mean latency miss: 4623 ms (n=61)
- mean latency near_hit: 3292 ms (n=18)
