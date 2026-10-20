# Architecture

How a request flows through GistCache, how the near-hit patcher works, how the cache is stored, scoped and invalidated, why it is built this way, and its full limitations, roadmap and test coverage. Back to the [README](../README.md).

## The problem

Applications built on LLMs pay for the same questions again and again in slightly different words. "How do I reset my password" and "forgot my password, help" trigger two full model calls even though one answer serves both.

Semantic caches address this by embedding each query and returning a cached answer when a new query is similar enough. Most of them use a single cosine similarity threshold. That design has two flaws.

**1. Wrong answers on lookalike queries.** Embedding models place these pairs very close together even though they mean different things:

| Query A | Query B | What changed |
|---|---|---|
| cancel my order | don't cancel my order | negation |
| flights to Toronto | flights from Toronto | direction |
| sort a list in Python 2 | sort a list in Python 3 | version |
| weather in Toronto | weather in Waterloo | entity |

A loose threshold returns wrong answers for these. A strict threshold rejects them but also rejects most true paraphrases, so the hit rate collapses.

**2. Near-misses are thrown away.** "Sort a list in Python" and "sort a list in Python in descending order" count as a miss. The application pays full price even though almost all of the answer already exists.

## Components

The flow diagram is in the [README](../README.md#how-it-works).

| Component | Default | Role |
|---|---|---|
| Proxy | FastAPI | Exposes an OpenAI-compatible `/v1/chat/completions` endpoint |
| Embedder | `all-MiniLM-L6-v2` (local) | Turns queries into vectors |
| Vector index | Exact cosine search (numpy) over embeddings stored in SQLite | Finds the top-k most similar cached queries |
| Verifier | Rule checks, a local cross-encoder, a local NLI contradiction guard and an optional small-model second opinion | Classifies each candidate as hit, near-hit or miss |
| Patcher | Configurable small model | Adapts a cached answer to the new query or refuses |
| Large model | Configurable large model | Answers misses and fallbacks |
| Store | SQLite | Holds entries, metadata and the decision log |

The embedder and cross-encoder run locally, so they add latency but no API cost.

## Request lifecycle

1. **Receive.** The application sends a normal chat completion request to GistCache.
2. **Bypass check.** Requests with tool definitions, a high temperature, an `X-GistCache-Bypass: true` header go straight to the large model.
3. **Normalize.** Lowercase, collapse whitespace and strip boilerplate. The original text is kept for the verifier.
4. **Fingerprint context.** For multi-turn requests, build a context key from recent turns (see [Multi-turn conversations](#multi-turn-conversations)).
5. **Retrieve.** Embed the query and fetch the top-k candidates (default k = 5) within the same scope and context key.
6. **Verify.** Run the rule checks, the cross-encoder and the safety layers (see [VERIFIER.md](VERIFIER.md)) on each candidate. Pick the best candidate and a tier.
7. **Serve.**
   - Hit: return the cached answer.
   - Near-hit: call the patcher. If it returns `CANNOT_PATCH` or fails validation, fall back to the large model.
   - Miss: call the large model.
8. **Store.** Cache every large model answer, unless it is time-sensitive or the optional volatility check says it goes stale. Patched answers are not cached by default so errors cannot compound.
9. **Log.** Record the tier, the reason, the candidate used, scores, latency and cost.

## The near-hit tier

The patcher is a small model with a strict job: adapt a cached answer to a new query **without inventing new facts.**

### Patcher input

- The original cached query
- The cached answer
- The new query
- The list of differences the verifier found

### Patcher rules

The prompt (`patcher.prompt: v2`, the default) tells the small model what it may and may not do:

- **May:** restructure what is already there (prose to a table, a list, JSON or a paragraph; reorder; change sort order), condense when asked to be brief, simplify wording using only ideas already present, and select the part of the answer that satisfies a narrowing constraint. None of this counts as inventing facts.
- **Must not:** add facts, steps, numbers, names, examples or code that are not in the cached answer, or change the meaning of anything it keeps.
- **Refuse:** reply exactly `CANNOT_PATCH` only when a correct answer would need content the cached answer lacks (a new example, a new comparison, extra detail). Presenting existing content differently is not new information.

The original, stricter prompt is kept as `v1` (`patcher.prompt: v1`). Prompt v1 refused pure reformatting requests such as "in a short table"; see the [patcher prompt A/B](EVALUATION.md#patcher-prompt-ab).

### Output validation

A patched answer is rejected and the request goes to the large model when one of the checks below fails. The request is then logged with tier `miss` and reason `miss: near_hit fallback: <REASON>` (for example `miss: near_hit fallback: CANNOT_PATCH`), and the response carries `X-GistCache-Fallback: true`:
- `CANNOT_PATCH`: the patcher refused.
- `EMPTY`: the output is empty.
- `TOO_LONG`: the output is longer than the larger of `max_length_ratio` times the cached answer or the cached length plus 150 characters. The slack exists because table, list and code-fence markup is a roughly fixed number of characters that dominates a short answer; a pure ratio rejected a correct 192-character table of a 113-character answer. On long answers the ratio is the larger term and still catches invented content.
- `TOO_SHORT`: the output is under a fifth of the cached answer, unless the request asks for something short or structured (brief, list, bullets, table, JSON, summary).

Patching is only worth it if it is actually cheaper. The evaluation counts the full cost of the patch call plus every fallback, not just the successful patches.

## Cache store

SQLite holds the entries and their embeddings (as a float32 `BLOB`). Retrieval loads the candidates for the request's scope and context key and ranks them by exact cosine similarity, which is fine up to tens of thousands of entries. A vector index (FAISS or pgvector) is on the [roadmap](../README.md#limitations-and-roadmap).

```sql
CREATE TABLE entries (
    id            TEXT PRIMARY KEY,
    scope         TEXT NOT NULL,      -- tenant or app namespace
    context_key   TEXT,               -- multi-turn fingerprint, NULL for single-turn
    query         TEXT NOT NULL,      -- original query
    query_norm    TEXT NOT NULL,      -- normalized query
    answer        TEXT NOT NULL,
    model         TEXT NOT NULL,      -- model that produced the answer
    created_at    INTEGER NOT NULL,
    expires_at    INTEGER,            -- NULL means no TTL
    hit_count     INTEGER DEFAULT 0,
    last_hit_at   INTEGER,
    embedding     BLOB                -- float32 vector of the normalized query
);

CREATE TABLE decisions (
    id            TEXT PRIMARY KEY,
    request_at    INTEGER NOT NULL,
    scope         TEXT NOT NULL,
    query         TEXT NOT NULL,
    tier          TEXT NOT NULL,      -- hit | near_hit | miss | bypass
    reason        TEXT NOT NULL,
    candidate_id  TEXT,
    cosine        REAL,
    cross_score   REAL,
    differences   TEXT,               -- JSON list
    fell_back     INTEGER DEFAULT 0,
    latency_ms    REAL NOT NULL,
    cost_usd      REAL NOT NULL
);
```

Eviction is least recently used once the store passes `max_entries`.

## Multi-turn conversations

"What about the second one?" means nothing without the conversation before it. Matching it against the cache on its own text would return a random answer.

GistCache handles this in two ways:
- **Context key.** For multi-turn requests, the last N turns are summarized into a short canonical string and hashed. Candidates must share the same context key.
- **Unresolved references.** Queries that depend on earlier turns ("it", "that", "the second one") with no matching context key are always a miss.

Single-turn requests have no context key and match only other single-turn entries.

## Invalidation and scoping

- **TTL.** Each entry can expire. The default is no TTL. Time-sensitive queries are never cached at all.
- **Scope.** Every entry belongs to a scope (an app, tenant or user ID passed in a header). Lookups never cross scopes, so one user's cached answer can never reach another user.
- **Model changes.** Entries record which model produced them, but changing the large model does not invalidate older entries automatically yet. Use `gistcache purge` after switching models.
- **Manual purge.** `gistcache purge --scope <name>` or `--older-than 7d` (units `s`, `m`, `h`, `d`).

## Explainability and logging

Every decision is logged with its reason, so you can see exactly why a request was served the way it was.

```bash
gistcache log --last 5
```

Example output from the real proxy and verifier models with mock LLM providers (the same session as the [README example](../README.md#example); latency and cost are near zero because no real model was called):

```
TIME      TIER      REASON                                   COS   XENC      MS COST
18:42:13  miss      miss: ['python'] -> ['javascript']       0.67  0.00      14 $0.0000
18:42:13  near_hit  near_hit: sort order                     0.89  0.79      14 $0.0000
18:42:13  hit       hit: paraphrase                          0.98  0.98     387 $0.0000
18:42:12  miss      miss: no candidates                      -     0.00      98 $0.0000
```

An illustrative example with the kind of values a real deployment produces (made-up values, real format). A near-hit whose patch is rejected is logged with tier `miss` and reason `miss: near_hit fallback: <REASON>`, and its response carries `X-GistCache-Fallback: true`:

```
TIME      TIER      REASON                                   COS   XENC      MS COST
03:12:01  hit       hit: paraphrase                          0.94  0.97      18 $0.0000
03:12:09  near_hit  near_hit: sort order                     0.91  0.83     410 $0.0004
03:12:15  miss      miss: negation mismatch                  0.93  0.21    1840 $0.0061
03:12:22  miss      miss: entity ['toronto'] -> ['waterloo'] 0.95  0.66    1720 $0.0058
03:12:30  miss      miss: near_hit fallback: CANNOT_PATCH    0.88  0.74    2210 $0.0065
```

`gistcache stats` prints, for each tier, the request count, total cost and mean latency over the whole decision log.

## Design decisions

**Why a cross-encoder instead of a better threshold?**
Cosine similarity compares two compressed vectors, so small but critical words like "not" barely move it. A cross-encoder reads both queries together and catches these differences. It is slower, so it only runs on the top-k candidates.

**Why rules on top of the cross-encoder?**
Rules are cheap, predictable and explainable. They catch the failure types that matter most (negation, numbers, entities) with certainty and give every decision a readable reason.

**Why can the patcher refuse?**
Forcing every similar query through adaptation produces confident wrong answers. Letting the patcher return `CANNOT_PATCH` and falling back to the large model keeps quality high. The evaluation counts the cost of every fallback so the savings stay honest.

**Why not cache patched answers?**
A patched answer was produced from another cached answer. Caching it would let small errors stack across generations. Only large model answers become cache entries by default.

**Why a contradiction guard that can only veto?**
As an equivalence check, NLI rejected true paraphrases ("tips for a job interview" is not a statement that entails another). As a contradiction detector it separated opposites ("attract / repel", "iphone / android phone") from paraphrases cleanly. Using it only as a veto keeps its strength and avoids its weakness.

**Why a second opinion only on borderline pairs?**
A model call on every candidate would add cost and latency to every hit. The cross-encoder is confident at the extremes and unsure in the middle, so only the middle is asked, after every free check has run.

**Why never cache time-sensitive queries?**
"What is the latest version" has a different correct answer next month. No similarity check can know that.

## Limitations

- **Generated variants.** Most variants in the main stream are generated by a model, so they may not match the distribution of real user rephrasing. The hard set and the manual spot check reduce this but do not remove it.
- **Replayed workload.** Cost savings come from a replayed query stream, not production traffic. Real repetition rates vary widely between applications.
- **Judge reliability.** LLM judges make mistakes. A human-reviewed sample with a kappa score, which would show how far to trust the judge, is planned but not done.
- **English only.** Rules and models are tuned for English.
- **Rule coverage.** Rules catch known difference types. New kinds of critical differences can still slip through to the cross-encoder.
- **Single node.** SQLite and in-process numpy search run on one machine. A shared deployment would need a networked store and vector index.
- **Rule vocabularies.** The antonym lexicon, constraint phrases and volatile-topic patterns are finite word lists. The contradiction guard and the second opinion exist because of that, but they are themselves models and can be wrong.
- **Small samples.** Every number comes from a few hundred hand-built pairs, 96-query pilots or one 596-query run on real prompts; the hand-built pairs were written by the same author as the rules, and variants and judge come from the same model family.
- **Latency on misses.** Every miss pays for the local verifier models and a synchronous volatility check, so the median request was slower than with no cache in the LMSYS evaluation.
- **Recall on real text.** The number and entity rules are conservative and blocked many safe reuses of real-user paraphrases; GistCache reused 43% of the reusable queries.
- **Judge noise.** Several rejected reuses were judge or reference disagreements on subjective or ambiguous questions, and answers were capped at 300 tokens. The same judge was applied to every system.
- **Patcher behaviour.** Near-hits save cost but little time, the small model sometimes refuses reformatting requests, and its output varies between runs.
- **Entities.** There is no NER model; entity swaps rely on a capitalization heuristic and the model-based layers.

## Roadmap

Done in v1:

- [x] Evaluation on LMSYS-Chat-1M prompts with a threshold-cache baseline (one run: 149 seeds, 596 queries)
- [x] LLM judging of every reused answer from every system, against a large-model reference
- [x] Contradiction guard, borderline second opinion and volatility check, each validated on small samples
- [x] Regression pins on all 245 labelled pairs, and CI

Remaining:

- [ ] Independent judge (not the patcher's model family), a human-checked sample with Cohen's kappa, and bootstrap confidence intervals
- [ ] Move the volatility check off the request path (store answers in the background)
- [ ] Fewer rule false positives on real text (number and entity rules), measured on a fresh real-prompt sample
- [ ] Faster near-hits (edit-based patching instead of rewriting the whole answer)
- [ ] NER for entities
- [ ] FAISS or pgvector index for large caches
- [ ] Automatic invalidation when the large model changes
- [ ] Adaptive thresholds learned per query cluster
- [ ] Feedback loop: user thumbs-down evicts an entry and adjusts thresholds
- [ ] Redis and pgvector backends for multi-node deployments
- [ ] Prometheus metrics endpoint and a small dashboard
- [ ] Multilingual rules and models

## Testing

```bash
pytest
```

- **Rule and policy tests** cover each difference check (including regressions found on real LLM-written text) and the tier decision policy.
- **Pipeline tests** run the full request lifecycle against mock providers: tiers, scope isolation, bypass paths, multi-turn context, the contradiction guard, the volatility check and decision logging.
- **Proxy integration tests** (`tests/test_server.py`) drive the FastAPI app with a test client and mock providers: the `X-GistCache-*` response headers, hit / near-hit / fallback / bypass behaviour, scope isolation, streaming, `/v1/models`, and the 502 returned when the upstream provider fails.
- **Pinned regression tests** (`tests/test_pinned.py`) record the verifier's decision on all 245 labelled eval pairs. The rule pins are fast and run on every `pytest`. The full-pipeline tier pins load the real models (about 10 seconds) and run with `GISTCACHE_SLOW=1 pytest`. If a behaviour change is intended, run `python -m eval.pin`, review the diff of `eval/pinned.json`, and commit it with the change.
- **Harness tests** cover the spend cap, stream ordering, variant parsing and judge verdict parsing.
- **CLI and config tests** (`tests/test_cli.py`) cover `log`, `stats`, `purge` (by scope and by age), `serve` wiring, and that every shipped config loads and builds its providers.
- **CI** runs the fast suite on every push (Python 3.10 and 3.12) without the heavy ML packages.
- **Not covered:** the real provider classes, which need network calls and a key, and the model-dependent behaviour outside the pinned eval pairs.
