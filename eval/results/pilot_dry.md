# GistCache pilot

32 queries run. API spend $0.0205 of $5 cap (large 32 calls $0.0196, small 40 calls $0.0009).

## Tier by query type

| Query type | n | Expected | hit | near_hit | miss | Over-reuse |
|---|---|---|---|---|---|---|
| seed | 8 | miss | 0 | 0 | 8 | 0 |
| paraphrase | 8 | hit | 8 | 0 | 0 | 0 |
| patchable | 8 | near_hit | 0 | 8 | 0 | 0 |
| critical | 8 | miss | 0 | 0 | 8 | 0 |

## Quality of reused answers (LLM judge vs fresh reference)

- hit: 8/8 acceptable (0 not judged)
- near_hit: 8/8 acceptable (0 not judged)

## Cost and latency

- GistCache pipeline cost: $0.0103
- No-cache estimate (32 x mean large call $0.00061): $0.0196
- Saved: 48%
- mean latency hit: 2 ms (n=8)
- mean latency miss: 1 ms (n=16)
- mean latency near_hit: 1 ms (n=8)
