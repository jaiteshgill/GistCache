# The verifier

How GistCache decides whether a cached answer can be reused: the rule tables, the three stages, threshold behaviour, the additional safety layers and the tests behind them. Back to the [README](../README.md).

## The verifier

The verifier is the core of GistCache. It answers one question: **how do these two queries differ, and is that difference safe to reuse across?**

### Stage 1: difference extraction (rules)

Fast deterministic checks on the original text of both queries.

| Check | Example | Classification |
|---|---|---|
| Negation | cancel vs don't cancel | Critical |
| Direction | to Toronto vs from Toronto | Critical |
| Entity swap | Toronto vs Waterloo | Critical (needs new facts) |
| Number, date or quantity change | 3 items vs 5 items | Critical |
| Language or version change | Python 2 vs Python 3, Java vs Rust | Critical |
| Time-sensitive terms | latest, today, current, price | Critical (never reused) |
| Output format | as a list vs as a table, return JSON | Patchable |
| Order or sort direction | ascending vs descending | Patchable |
| Verbosity or tone | briefly, explain like I'm five | Patchable |
| Narrowing constraint | add comments, use type hints | Patchable |

Numbers, negations, directional words, languages and time-sensitive terms use curated patterns. Entities are detected with a heuristic (capitalized words that do not start a sentence) plus a comparison of the remaining content words; there is no NER model yet. Lowercase entity swaps ("best laptop for students" vs "for gamers") are therefore caught by the cross-encoder, the contradiction guard and the second opinion, not by a rule.

### Stage 2: semantic scoring (cross-encoder)

A cross-encoder reads both queries together and scores how closely they match. It is much more accurate than cosine similarity on lookalike pairs because it compares the queries word by word instead of comparing two compressed vectors.

### Stage 3: decision policy

```
cosine      = similarity of the two query embeddings (retrieval)
cross_score = cross-encoder score for the pair (duplicate detection)

if any critical difference:                                    -> MISS
elif no differences and cross_score >= hit_threshold:          -> HIT
elif only patchable differences and cosine >= near_threshold:  -> NEAR-HIT
else:                                                          -> MISS
```

Near-hits are gated on cosine similarity, not on the cross-encoder score: the cross-encoder is trained to detect duplicate questions, so it correctly scores "sort ascending" against "sort descending" near zero, and the rules have already explained that difference. The cross-encoder score gates hits only.

Every decision carries a reason string such as `miss: negation mismatch` or `near_hit: sort order`. The defaults are `hit_threshold` 0.90 and `near_threshold` 0.70. A sweep of the hit threshold from 0.6 to 0.9 on the development pairs was flat, so there is no evidence for moving them; they have not been tuned on any test split.

### Additional safety layers

Evaluation on fresh data showed that rule lists and a duplicate-detection cross-encoder miss unseen antonyms, same-slot entity swaps and volatile topics. Four layers were added:

- **Content safety net** (`verifier.strict_content_check`). If one query's content words are a strict subset of the other's, the extra words are content the cached answer was not written for, so the pair is downgraded from hit to near-hit. A related mark, `content_substituted`, is added when a near-hit's patchable difference ("short" vs "brief") sits next to a swapped content word ("brave" vs "cowardly"), so the swap is not hidden behind the harmless difference. Patchable phrases (format, order, verbosity, constraints) are removed before these comparisons. The net costs some hits and buys safety.
- **Contradiction guard** (`verifier.nli_guard`). An NLI model checks whether the two queries contradict each other and vetoes the reuse if so. It runs on hits, on near-hits that exist only because of the content net, and on near-hits with a hidden content swap. Rules cannot list every antonym ("attract / repel"); a contradiction model generalizes. Used as an equivalence check NLI performed poorly, so the guard only vetoes.
- **Volatility check** (`cache.volatility_check`). When a large-model answer is about to be stored, the small model is asked once whether the correct answer changes over time (prices, scores, office-holders, "is it open"). If so, the answer is not cached. No similarity check can detect this, because "is it raining in Vancouver?" does not contradict itself. Measured with Claude Haiku 5.5: 14 of 14 volatile and 0 of 14 stable queries flagged (`python -m eval.volatility`).
- **Borderline second opinion** (`verifier.borderline_judge`, off by default, on in `configs/anthropic.yaml`). The cross-encoder rejects many true paraphrases in the middle of its score range. For pairs that no rule objects to, the contradiction guard clears, and whose score falls between `borderline_low` and `hit_threshold`, the small model is asked whether one answer would serve both queries. A YES promotes the pair to a hit; the guard still has the last word, and the free local checks run first, so a vetoed pair costs no API call. Measured with Claude Haiku 5.5 on small samples: of the 9 pairs in the borderline band across the eval sets it recovered 8 of 8 true paraphrases and rejected the 1 near-miss ("headache" vs "migraine"); in a separate stress test (`python -m eval.borderline_stress`) it said NO to all 24 near-misses (different audience, platform, scope, entity or condition) and YES to 11 of 12 fresh paraphrases. The one failure was the safe kind (a lost hit). These samples are small and were written by the same author as the prompt, so treat them as encouraging, not conclusive. `python -m eval.borderline` lists, for free, the pairs that would be asked.

Rule improvements made after testing on real LLM-written text: negation words are treated as one signal (`no`, `not`, `non-`, `without`), prefix negation and an antonym lexicon are checked, spelled-out numbers and ordinals count as numbers, possessives are stripped ("Java's"), sentence-initial capitals are not entities, `go`/`c`/`r` count as languages only in a language context, "A and B" / "A or B" comparisons are symmetric, direction swaps require shared vocabulary, and "weather" is time-sensitive only when asking about conditions.

## Verifier tests

The hand-written labelled pairs below came before the real-prompt evaluation and drove most of the rule and safety-layer changes described above. The end-to-end pilots, the patcher A/B and the real-prompt results are in [EVALUATION.md](EVALUATION.md).

**Verifier on hand-written pairs** (`python -m eval.run --file <set>.jsonl --split all`; real embedder, cross-encoder and NLI guard). "Wrongly reused" counts pairs where GistCache served more than the expected tier.

| Set | Pairs | Plain threshold cache correct | GistCache correct | Wrongly reused |
|---|---|---|---|---|
| `hard_set.jsonl` (rules were tuned on this) | 118 | 50 | 113 | 0 |
| `held_out.jsonl` (rules were tuned on this too) | 58 | 30 | 54 | 0 |
| `fresh.jsonl` (written after tuning, not tuned on) | 69 | 36 | 55 | 6 (all volatile queries) |

The fresh set is the honest figure. Before the contradiction guard it was 50/69 with 11 wrongly reused. The 6 volatile queries are addressed by the volatility check, which caught 14 of 14 volatile and 0 of 14 stable queries on a separate 28-query check (`python -m eval.volatility`). Every set has been read by the author, so none is a clean held-out set.

Edge cases these sets cover beyond the [hard test set categories](EVALUATION.md#hard-test-set-categories): antonyms with no negation word, spelled-out numbers and ordinals, lowercase entities, role swaps ("Google sued Oracle" / "Oracle sued Google"), symmetric comparisons ("A or B" / "B or A"), C versus C++, and volatile questions phrased without trigger words. The test files are `eval/hard_set.jsonl`, `eval/held_out.jsonl` and `eval/fresh.jsonl`; `tests/test_pinned.py` pins the verifier's decision on every one of these 245 pairs.
