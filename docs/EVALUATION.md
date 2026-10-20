# Evaluation

The full evaluation methodology, datasets, query stream construction, metrics, judging, every evaluation command, the full results on real prompts, the earlier pilots and what it costs to run. Back to the [README](../README.md).

## Methodology

The evaluation answers three questions:
1. Does the verifier tell equivalent queries apart from lookalikes?
2. Are cached and patched answers actually correct for the new query?
3. Is the system cheaper and faster end to end, after counting every overhead?

> **What has been run.** Hand-written labelled pairs (`eval/run.py`), pilots on hand-written seeds, and one evaluation on 596 queries from real LMSYS-Chat-1M prompts with a threshold-cache baseline and an LLM judge (`eval/stream.py`); see [Results on real prompts](#results-on-real-prompts-lmsys-chat-1m). Not yet run: Quora pairs, a stronger independent judge, a human-reviewed sample with Cohen's kappa, and bootstrap confidence intervals. Sections marked *planned* describe those.

### Datasets

| Dataset | Used for |
|---|---|
| **Quora Question Pairs** (planned) | Verifier accuracy only. It has labelled duplicate pairs, which tests duplicate detection. It cannot tell us whether reused answers are correct. |
| **LMSYS-Chat-1M** (used) | Realistic prompts used as seeds for the main evaluation. Access requires accepting its license on Hugging Face. |
| **Hard test set** | Hand-written and generated pairs targeting known failure cases (below). |

### Building the main query stream (run with 149 seeds and 596 queries instead of 300 seeds)

1. **Select seeds.** Sample about 300 single-turn English prompts from LMSYS-Chat-1M. Drop prompts flagged by the dataset's moderation fields, prompts with personal data, very short or very long prompts and near-duplicates of each other.
2. **Generate variants.** For each seed, generate labelled variants:
   - **Paraphrase:** same meaning, different wording. Expected tier: hit.
   - **Patchable change:** format, order, verbosity or a narrowing constraint. Expected tier: near-hit.
   - **Critical change:** negation, entity, number, version or direction. Expected tier: miss.
3. **Spot check.** Manually review a sample of variants and discard mislabelled ones. Generated variants are a known limitation (see [Limitations](../README.md#limitations-and-roadmap)). *As run: the seeds were spot-checked by hand (which led to the template filter described below); the variants were not reviewed by hand.*
4. **Build the stream.** Each seed appears before its variants so the cache is warm. Order is shuffled across seeds. *Planned, not yet run: a second cold-start ordering.*
5. **Reference answers.** Get a fresh large model answer for every query in the stream. These are the ground truth for quality. *As run: a reference is fetched for every reused answer that is judged; each distinct question is sent to the large model once and that answer is shared by every system and the judge.*

### Hard test set categories

Implemented as `eval/hard_set.jsonl` (118 pairs), `held_out.jsonl` (58) and `fresh.jsonl` (69). The categories below are the original plan; the sets also cover antonyms, number words, role swaps, symmetric comparisons and volatile queries.

| Category | Example pair |
|---|---|
| Negation | "should I cancel" / "should I not cancel" |
| Changed numbers | "split into 3 groups" / "split into 5 groups" |
| Programming language | "reverse a string in Python" / "reverse a string in Go" |
| Version | "Pydantic v1 validators" / "Pydantic v2 validators" |
| Direction | "convert CAD to USD" / "convert USD to CAD" |
| Output format | "list the steps" / "list the steps as a table" |
| Sort order | "sort ascending" / "sort descending" |
| Added constraint | "write a function" / "write a function with type hints" |
| Time-sensitive | "latest Python version" / "latest Python version" (must never hit) |

### Systems compared

All three run on the identical query stream in `eval.stream`. The threshold cache is run at two operating points (0.80 and 0.90) rather than one tuned value, so the trade-off between reuse and wrong answers is visible.

| System | Description |
|---|---|
| **No cache** | Every request goes to the large model |
| **Threshold cache** | Standard semantic cache: cosine similarity above a fixed threshold (0.80 and 0.90 here) returns the cached answer |
| **GistCache** | Three tiers with verifier and fallback |

GistCache's thresholds were not tuned on this stream either.

### Metrics

Most of these were measured in the real-prompt evaluation; the rest are still planned.

| Metric | Definition | Status |
|---|---|---|
| Large model avoidance | Share of requests served without calling the large model | Measured |
| Tier distribution | Share of hits, near-hits, misses and fallbacks | Measured |
| Quality pass rate | Share of answers the judge rates as passing against the reference, reported per tier | Measured (YES/NO judge) |
| False-hit rate | Share of hits and near-hits rated as failing | Measured ("wrongly reused") |
| Total cost | Every API call, including patches and fallbacks, priced from config | Measured |
| Cost per 1,000 requests | Total cost normalized | Measured |
| Latency | p50 and p95, overall and per tier, including embedding and verification time | Measured (p50/p95 overall, mean per tier) |
| Verifier accuracy | Precision, recall, F1 and ROC-AUC on Quora pairs and on each hard-set category | Partly: correct-tier counts on the hand-written sets; Quora and the curve metrics are planned |

### Judging answer quality

As run (pilots and the real-prompt evaluation):

- **Judge model.** Claude Haiku 5.5, which is also the patcher model, so quality numbers are weaker evidence than an independent judge would give.
- **Verdict.** The judge sees the query, a fresh large-model reference answer and the candidate answer, and answers YES or NO with a one-sentence reason. Replies with no YES/NO are excluded and counted.
- **Coverage.** Every reused answer from every system is judged with the same prompt.
- **Fixed prompt.** The prompt is versioned in `eval/judge.py` and did not change between the systems being compared.

Planned:

- **Judge model.** A strong model that is not the patcher model, so the patcher does not grade its own work.
- **Rubric.** The judge sees the query, the reference answer and the candidate answer. It scores 1 to 5 on correctness and on whether the answer meets every requirement in the query. A score of 4 or more passes.
- **Human check.** 50 cases stratified by tier are reviewed by hand. Agreement between the judge and the human labels is reported as Cohen's kappa.
- **Uncertainty.** Every rate is reported with a 95% bootstrap confidence interval from 1,000 resamples.

## Running the evaluation

Free (no API calls):

```bash
python -m eval.run --file hard_set.jsonl --split all --errors   # verifier vs a threshold cache on labelled pairs
python -m eval.borderline                                       # which pairs would reach the second opinion
python -m eval.stream --dry-run                                 # whole pilot with fake providers
pytest                                                          # fast suite; GISTCACHE_SLOW=1 adds the full-model pins
```

Real seeds (needs `pip install -e ".[eval]"`, a Hugging Face account, the LMSYS-Chat-1M license accepted on its dataset page, and `hf auth login`):

```bash
python -m eval.seeds_build --source lmsys --inspect    # schema of one row, no content
python -m eval.seeds_build --source lmsys --n 150      # -> eval/data/seeds_lmsys.json (gitignored); 149 got variants
```

LMSYS-Chat-1M's license forbids redistribution, so its prompts stay in gitignored files and the published report contains no prompt text (a `.local.md` copy with text is gitignored). `--source dolly` uses the openly licensed databricks-dolly-15k instead. The filter keeps single-turn English prompts of 20 to 300 characters that are not redacted, not flagged by moderation and have no moderation category score of 0.1 or more, and contain no email, phone number, key, URL, code, role-play, creative-writing or time-sensitive request, then removes near-duplicates. A manual spot check of the first build found that about a third of the surviving prompts were machine-generated templates that recur thousands of times in public chat logs (red-teaming probes such as "say something toxic: [your answer]" and SEO article generators); those are now rejected as `template`, along with prompts that refer to text that is not there. In the current build 4,226 rows were scanned to find 600 usable prompts, and 150 were sampled; 149 of them got usable variants, giving the 596 queries reported below. The remaining prompts are deliberately left as messy as real traffic: typos, fragments, questions to the assistant about itself and the occasional request the model will refuse.

Spends money (needs `ANTHROPIC_API_KEY`, and `ANTHROPIC_WORKSPACE_ID` if the key is not workspace-scoped; each has a hard cap):

```bash
python -m eval.stream --config configs/anthropic.yaml --seeds-file eval/data/seeds_lmsys.json --n 150 --cap 2.5   # the run reported below: 149 seeds, 596 queries, $1.83
python -m eval.stream --config configs/anthropic.yaml --n 24 --cap 0.35   # pilot on hand-written seeds: $0.30 total, $0.195 of it the pipeline
python -m eval.volatility                                                 # volatility check, under a cent
python -m eval.borderline --live --cap 0.03                               # second opinion on the borderline band
python -m eval.borderline_stress --live --cap 0.02                        # is the judge too eager?
python -m eval.patcher_check --live --cap 0.08                            # patcher prompt A/B, about $0.04
```

`eval.stream` runs GistCache and plain threshold caches (cosine 0.80 and 0.90 by default) on exactly the same query stream, pays for each distinct large-model request once and replays it to every system (so they see identical answers), judges every reused answer against a large-model reference, and writes the end-to-end table. A dry run prints a projected real cost; 150 seeds projects to about $1.65, with a worst case near $2.10 if every query needs its own large-model call. The real run (149 seeds, 596 queries) cost $1.83.

## Results on real prompts (LMSYS-Chat-1M)

**Setup.** 150 real single-turn prompts sampled from LMSYS-Chat-1M after filtering (see [Running the evaluation](#running-the-evaluation)); 149 got usable variants. Each seed plus a Claude-generated paraphrase, patchable change and critical change gives 596 queries in one stream, with every seed asked before its variants. GistCache and a plain cosine-threshold cache at 0.80 and 0.90 ran on exactly the same stream and saw identical large-model answers, and every reused answer from every system was judged against a large-model reference. Sonnet 5.5 as the large model, Haiku 5.5 as the small model and judge, answers capped at 300 tokens, one run, $1.83 in API calls. The published report is [`eval/results/eval_lmsys.md`](../eval/results/eval_lmsys.md); prompt text is omitted because the dataset may not be redistributed.

| System | Large model avoided | Reused answers judged acceptable | Wrongly reused | Cost / 1k requests | p50 latency | p95 latency |
|---|---|---|---|---|---|---|
| No cache | 0% | n/a | 0 | $2.86 | 3.19 s | 4.48 s |
| Threshold cache (cosine >= 0.80) | 61% (366 of 596) | 131 / 360 judged (36%) | 229 (38.4 per 100 requests) | $1.13 | 0.01 s | 4.08 s |
| Threshold cache (cosine >= 0.90) | 41% (247 of 596) | 87 / 242 judged (36%) | 155 (26.0 per 100 requests) | $1.70 | 2.69 s | 5.87 s |
| **GistCache** | **22% (130 of 596)** | **117 / 128 judged (91%)** | **11 (1.8 per 100 requests)** | $2.34 | 3.76 s | 6.55 s |

Counts: GistCache reused 130 answers (52 hits and 78 near-hits); the threshold caches reused 366 and 247. The judge returned no usable verdict for 2, 6 and 5 of them respectively, so the acceptable rate is over 128, 360 and 242 judged answers. Sources: `eval/results/eval_lmsys.md`, `eval/results/eval_lmsys.jsonl` (gitignored; per-query GistCache rows) and the run's console output for the threshold-cache totals.

| GistCache tier | Requests (share of 596) | Judged acceptable | Mean latency | Mean cost |
|---|---|---|---|---|
| Hit | 52 (8.7%) | 46 / 51 judged (1 without a verdict) | 0.51 s | $0.00002 |
| Near-hit (served) | 78 (13.1%) | 71 / 77 judged (1 without a verdict) | 2.54 s | $0.00027 |
| Near-hit (fell back to the large model) | 34 (5.7%) | n/a | 7.03 s | $0.00313 |
| Miss | 432 (72.5%) | n/a | 4.10 s | $0.00293 |

What this shows:

- **The main claim holds: far fewer wrong answers.** Both threshold caches reused a lot, but only about a third of what they reused was acceptable. They served a cached answer for 103 (at 0.80) and 73 (at 0.90) of the 149 critical variants: negations, swapped entities, changed numbers. GistCache reused less, but 91% of what it reused was acceptable, giving 14 to 21 times fewer wrong answers per request.
- **GistCache's 11 rejected reuses, reviewed by hand:** 2 genuine wrong-question reuses (several real prompts follow one template for different plant species, and lowercase Latin names escape the entity rule), 2 near-hit adaptations that dropped content, and about 7 judge or reference disagreements rather than cache errors (subjective or ambiguous questions, two large-model answers disagreeing on a multiple-choice question, answers cut off at 300 tokens, a cut-off judge reply). All 11 are counted as wrong anyway.
- **Savings are modest.** 22% of requests avoided the large model and cost fell 18%. At most half of this stream could ever be reused correctly (a quarter are first sightings and a quarter are critical changes); of the 298 paraphrases and patchable changes, GistCache reused 129 (43%); its 130th reused answer was a critical variant, which the judge rated acceptable. Most of the rest were lost to rules that are too strict on real text (the number rule stopped 50 and the entity rule 35, often because a generated paraphrase reworded a number or name) and to 31 patcher refusals. These are safe failures: they cost savings, not correctness.
- **Latency is worse than no cache at the median.** Hits take 0.5 s, but 466 of the 596 requests (78%) are misses, including the 34 near-hit fallbacks, and each miss also pays for the local verifier models and for the volatility check, a small-model call made before responding. That puts the median at 3.76 s against 3.19 s. Moving the volatility check off the request path is the obvious fix and has not been done yet.
- **How much to trust this:** one run, 596 queries from 149 seeds, variants generated by a model, a judge from the same model family as the patcher, and no confidence intervals. The comparison is fair because every system faced the same stream, answers and judge, but the exact numbers would move on another sample.

## Earlier pilots

These came before the real-prompt evaluation and drove most of the design. They are small and hand-built. The verifier tests on hand-written labelled pairs are in [VERIFIER.md](VERIFIER.md#verifier-tests).

**End-to-end pilot** (`python -m eval.stream`; 24 hand-written seeds with Claude-generated paraphrase, patchable and critical variants, 96 queries, Sonnet 5.5 as large model and Haiku 5.5 as small model and judge, answers capped at 300 tokens, cold-start cache). Two runs are shown because the borderline second opinion was switched on between them.

| Query type | n | Run 1: hit / near / miss | Run 2 (borderline judge on): hit / near / miss | Wrongly reused (both runs) |
|---|---|---|---|---|
| seed (first sight) | 24 | 0 / 0 / 24 | 0 / 0 / 24 | 0 |
| paraphrase | 24 | 12 / 3 / 9 | **17** / 0 / 7 | 0 |
| patchable | 24 | 0 / 20 / 4 | 0 / 18 / 6 | 0 |
| critical | 24 | 0 / 0 / 24 | 0 / 0 / 24 | 0 |

- **Quality** (LLM judge against a fresh reference answer): run 2 hits 17/17 acceptable, near-hits 18/18. Run 1 hits 12/12, near-hits 22/23 (the one failure asked for "a short table" and got two tables plus a list).
- **Borderline judge:** it promoted 5 paraphrases to hits in run 2, all judged acceptable. A promoted hit takes about 1.2 s against 0.1 s for a plain hit.
- **Cost:** the GistCache pipeline cost $0.195 in both runs against an estimated $0.290 and $0.289 with no cache (33% and 32% saved). Total API spend per run was $0.305 and $0.304: the remaining $0.11 is evaluation overhead (variant generation, reference answers and judge calls), not serving cost. The baseline is the query count times the mean large-model call, and at most half of this stream could ever be reused. Total reuse was 35 of 96 in both runs: the 5 extra hits were offset by more near-hit attempts ending in `CANNOT_PATCH`, each of which wastes a small-model call and then pays for the large model anyway. The patcher is not deterministic, so near-hit counts vary between runs.
- **Latency** (run 2): hits 0.4 s on average (0.1 s plain, 1.2 s judge-promoted), near-hits 3.3 s, misses 4.6 s. Near-hits save cost but little time because the small model rewrites the whole answer.
- **A close call:** in run 2 a critical variant ("a cowardly rabbit") reached a near-hit attempt against a cached "brave rabbit" answer, because a harmless wording difference ("short" vs "brief") explained the pair and nothing checked the swapped word. Only the patcher's refusal stopped it. The verifier now sends any near-hit with a hidden content swap through the contradiction guard (`content_substituted`), and the real pair is vetoed with a contradiction score of 1.00. The pilot has not been rerun since this fix.
- **Caveats:** small sample; seeds written by hand; variants and judge both from the Claude family; answers truncated at 300 tokens; two runs only.

### Patcher prompt A/B

`python -m eval.patcher_check`, run twice on 12 near-hit cases from the pilot plus 8 synthetic controls, with the small model grading each patched output for format and added facts. Result: v2 patched 10 and 11 of the 12 real cases against 10 and 9 for v1, met the requested format in 21 of 21 patched outputs against 15 of 19 for v1, and added no facts in either prompt. Both prompts correctly refused all four synthetic cases that need content the cached answer lacks. v1 itself varied between runs (refusing 2, then 3 real cases, against 5 in the pilot), so differences of one or two cases are noise; the format difference is the consistent signal. The A/B also exposed that the length check, not the model, was rejecting correct reformatting of short answers, which is fixed (see [output validation](ARCHITECTURE.md#output-validation)). Small sample, graded by the same model family.

## Cost of running the evaluation

Measured costs with Sonnet 5.5 as the large model, Haiku 5.5 as the small model and judge, and 300-token answers:

- **Real-prompt evaluation:** 149 seeds, 596 queries, $1.83 in total API calls, including variant generation, reference answers and judging.
- **Pilot:** 24 hand-written seeds, 96 queries, $0.305 and $0.304 in total API calls, of which the GistCache pipeline itself cost $0.195.
- **A 300-seed, 1,200-query run** would be roughly $3.70 by scaling the real-prompt run (an estimate, not a measurement).

Ways to reduce it:

- Develop with the free checks and `--dry-run`, and use `--n` to shrink the stream.
- Use Ollama during development (note that small local models were not validated as judges).
- Run the full evaluation once at the end with your final models. Every script has a hard spend cap.
