# GistCache pilot

5 queries run (stopped early: budget). API spend $0.0032 of $0.004 cap (large 5 calls $0.0031, small 5 calls $0.0000).

## Tier by query type

| Query type | n | Expected | hit | near_hit | miss | Over-reuse |
|---|---|---|---|---|---|---|
| seed | 5 | miss | 0 | 0 | 5 | 0 |

## Quality of reused answers (LLM judge vs fresh reference)

- hit: 0/0 acceptable (0 not judged)
- near_hit: 0/0 acceptable (0 not judged)

## Cost and latency

- GistCache pipeline cost: $0.0032
- No-cache estimate (5 x mean large call $0.00063): $0.0031
- Saved: -1%
- mean latency miss: 1 ms (n=5)
