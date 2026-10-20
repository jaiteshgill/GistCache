# GistCache pilot

96 queries run. API spend $0.3033 of $0.45 cap (large 96 calls $0.2906, small 139 calls $0.0127).

## Tier by query type

| Query type | n | Expected | hit | near_hit | miss | Over-reuse |
|---|---|---|---|---|---|---|
| seed | 24 | miss | 0 | 0 | 24 | 0 |
| paraphrase | 24 | hit | 9 | 3 | 12 | 0 |
| patchable | 24 | near_hit | 0 | 11 | 13 | 0 |
| critical | 24 | miss | 0 | 1 | 23 | 1 |

## Quality of reused answers (LLM judge vs fresh reference)

- hit: 1/9 acceptable (0 not judged)
- near_hit: 8/15 acceptable (0 not judged)
  - NOT acceptable [near_hit, paraphrase]: 'Give a short overview of how chess is played, covering its rules.' (served from near_hit: verbosity)
  - NOT acceptable [hit, paraphrase]: 'What is the reason Earth experiences different seasons?' (served from hit: paraphrase)
  - NOT acceptable [hit, paraphrase]: 'What steps should I follow to set up a virtualenv for Python?' (served from hit: paraphrase)
  - NOT acceptable [hit, paraphrase]: 'Can you explain what the Pythagorean theorem states, and how people apply it?' (served from hit: paraphrase)
  - NOT acceptable [near_hit, critical]: 'How do I structure an essay with a weak thesis?' (served from near_hit: content removed)
  - NOT acceptable [near_hit, patchable]: 'In a short table, explain how compound interest works.' (served from near_hit: output format)
  - NOT acceptable [hit, paraphrase]: 'What steps should I take to get ready for a job interview that focuses on behavioral questions?' (served from hit: paraphrase)
  - NOT acceptable [near_hit, patchable]: 'Explain how a hash map works, briefly and as a bulleted list.' (served from near_hit: output format)
  - NOT acceptable [near_hit, patchable]: 'Briefly, in a short bulleted list, how does garbage collection work in Java?' (served from near_hit: verbosity)
  - NOT acceptable [near_hit, patchable]: 'Explain the rules of chess briefly, as a bulleted list.' (served from near_hit: output format)
  - NOT acceptable [hit, paraphrase]: 'What steps can I take to raise my credit score?' (served from hit: paraphrase)
  - NOT acceptable [hit, paraphrase]: 'What are some ways I can overcome the habit of delaying work on large projects?' (served from hit: paraphrase)
  - NOT acceptable [hit, paraphrase]: 'What is a fair way to divide rent among roommates?' (served from hit: paraphrase)
  - NOT acceptable [near_hit, paraphrase]: 'What is the best way to organize an essay so that it has a powerful thesis statement?' (served from near_hit: verbosity)
  - NOT acceptable [hit, paraphrase]: 'What is the way to make a div sit in the middle of its container both horizontally and vertically using CSS?' (served from hit: paraphrase)

## Cost and latency

- GistCache pipeline cost: $0.2288
- No-cache estimate (96 x mean large call $0.00303): $0.2906
- Saved: 21%
- mean latency hit: 151 ms (n=9)
- mean latency miss: 4817 ms (n=72)
- mean latency near_hit: 4104 ms (n=15)
