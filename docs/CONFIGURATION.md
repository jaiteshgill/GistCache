# Configuration

Every configuration option, the shipped config files, and the HTTP API GistCache exposes, including all request and response headers. Back to the [README](../README.md).

## The example file

This is `config.example.yaml`. Every key shown is read by the code; there are no per-rule toggles.

```yaml
providers:
  large:
    kind: openai            # openai | anthropic | ollama | openai_compatible
    model: your-large-model
    api_key_env: OPENAI_API_KEY
    base_url: https://api.openai.com/v1
  small:
    kind: openai
    model: your-small-model
    api_key_env: OPENAI_API_KEY
    base_url: https://api.openai.com/v1

embedder:
  model: sentence-transformers/all-MiniLM-L6-v2
  device: cpu

verifier:
  cross_encoder: cross-encoder/quora-distilroberta-base
  hit_threshold: 0.90
  near_threshold: 0.70
  top_k: 5
  strict_content_check: true   # refuse a plain hit when one query adds/drops content no rule explains
  nli_guard: cross-encoder/nli-deberta-v3-small   # veto hits whose queries contradict each other
  contradiction_threshold: 0.5
  borderline_judge: true       # matches the reported run: adds one small-model judge call when the cross-encoder
                               # is unsure (score between borderline_low and hit_threshold) and no rule objects
  borderline_low: 0.3          # lowest cross-encoder score that is still worth asking about

patcher:
  enabled: true
  max_length_ratio: 1.5
  cache_patched_answers: false
  prompt: v2                   # v2 = pure reformatting is allowed (default); v1 = the strict original

cache:
  path: ./data/gistcache.db
  max_entries: 100000
  default_ttl_seconds: null
  volatility_check: true       # ask the small model once per stored answer if it goes stale (on in the reported
                               # evaluation; validated with Claude Haiku, re-run `python -m eval.volatility` for another model)

bypass:
  max_temperature: 0.7
  with_tools: true

multi_turn:
  context_turns: 4

pricing:                     # USD per 1M tokens
  large: { input: 0.0, output: 0.0 }
  small: { input: 0.0, output: 0.0 }
```

Fill in `pricing` with current rates from your provider so the cost numbers are accurate. `configs/anthropic.yaml` already has Anthropic's rates (checked 2026-10-20) and `configs/ollama.yaml` is all zeros.

## Every option

| Key | Default | Meaning |
|---|---|---|
| `providers.large`, `providers.small` | none (required) | The model that answers misses, and the model that patches near-hits and runs the optional judges. `small` may be omitted, which disables patching, the volatility check and the second opinion. |
| `…kind` | none | `openai`, `anthropic`, `ollama` or `openai_compatible`. Ollama and OpenAI-compatible servers use the OpenAI wire format. |
| `…model` | none | Model name sent to the provider. |
| `…api_key_env` | none | Name of the environment variable that holds the key. Ollama needs none. |
| `…base_url` | `https://api.openai.com/v1` (openai), `http://localhost:11434/v1` (ollama), `https://api.anthropic.com` (anthropic) | Override for proxies and self-hosted servers; required for `openai_compatible`. |
| `…workspace_id_env` | `ANTHROPIC_WORKSPACE_ID` | Anthropic only. If that variable is set, its value is sent as the `anthropic-workspace-id` header, which keys that are not tied to a workspace require. |
| `embedder.model` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model used for retrieval. |
| `embedder.device` | `cpu` | Device for the embedder. |
| `verifier.cross_encoder` | `cross-encoder/quora-distilroberta-base` | Local duplicate-question scorer that gates hits. |
| `verifier.hit_threshold` | `0.90` | Minimum cross-encoder score for a hit when no rule found a difference. |
| `verifier.near_threshold` | `0.70` | Minimum cosine similarity for a near-hit, and the minimum cosine for the second opinion to be asked. |
| `verifier.top_k` | `5` | Candidates retrieved and verified per request. |
| `verifier.strict_content_check` | `true` | The content safety net: a pair where one query adds or drops content no rule explains becomes a near-hit, not a hit. |
| `verifier.nli_guard` | `cross-encoder/nli-deberta-v3-small` | Local NLI model for the contradiction guard. Empty disables the guard. |
| `verifier.contradiction_threshold` | `0.5` | Contradiction probability at which the guard vetoes reuse. |
| `verifier.borderline_judge` | `false` (code), `true` (`config.example.yaml` and `configs/anthropic.yaml`) | Ask the small model about pairs the cross-encoder is unsure about. Adds one small-model call per borderline candidate. |
| `verifier.borderline_low` | `0.3` | Lowest cross-encoder score that is still worth a second opinion. |
| `patcher.enabled` | `true` | Off turns every near-hit into a miss. |
| `patcher.max_length_ratio` | `1.5` | Longest acceptable patch as a multiple of the cached answer (with 150 characters of slack for markup; see [output validation](ARCHITECTURE.md#output-validation)). |
| `patcher.cache_patched_answers` | `false` | Store patched answers as new entries. Off by default so errors cannot compound. |
| `patcher.prompt` | `v2` | `v2` allows pure reformatting; `v1` is the stricter original. |
| `cache.path` | `./data/gistcache.db` | SQLite file for entries and the decision log. `:memory:` keeps nothing on disk. |
| `cache.max_entries` | `100000` | Least recently used entries are evicted beyond this. |
| `cache.default_ttl_seconds` | `null` | Expiry for every new entry; `null` means none. |
| `cache.volatility_check` | `false` (code), `true` (`config.example.yaml` and `configs/anthropic.yaml`) | Ask the small model once per stored answer whether it goes stale; if so it is not cached. Adds one small-model call to every miss. |
| `bypass.max_temperature` | `0.7` | Requests with a higher `temperature` skip the cache. |
| `bypass.with_tools` | `true` | Requests with `tools` or `functions` skip the cache. |
| `multi_turn.context_turns` | `4` | How many earlier turns form the multi-turn context key. |
| `pricing.large`, `pricing.small` | `0.0` | USD per million input and output tokens, used only for cost accounting and the decision log. |

## Shipped configs

The evaluation reported in the README ran with `configs/anthropic.yaml`: patcher prompt `v2`, volatility check on, borderline second opinion on.

| File | Large / small model | Optional judges | Pricing |
|---|---|---|---|
| `config.example.yaml` | placeholders, OpenAI wire format | volatility check and second opinion on, as in the reported run | zeros, fill in |
| `configs/anthropic.yaml` | Claude Sonnet 5.5 / Claude Haiku 5.5 | volatility check and second opinion on (validated with Haiku) | Anthropic rates checked 2026-10-20 |
| `configs/ollama.yaml` | `llama3.1:8b` / `qwen2.5:3b` (local, free) | off: small local models were not validated as judges | zeros |

## API compatibility

| Feature | Supported |
|---|---|
| `POST /v1/chat/completions` | Yes |
| Streaming (`stream: true`) | Yes. Hits and near-hits are streamed from the final text. |
| Tool and function calling | Passed through to the large model, never cached |
| `GET /v1/models` | Yes |
| Embeddings and other endpoints | Not proxied in v1 |

### Response headers

| Header | Example |
|---|---|
| `X-GistCache-Tier` | `hit`, `near_hit`, `miss`, `bypass` |
| `X-GistCache-Reason` | `near_hit: sort order` |
| `X-GistCache-Source` | ID of the cached entry used |
| `X-GistCache-Fallback` | `true` if a near-hit fell back to the large model |

### Request headers

| Header | Effect |
|---|---|
| `X-GistCache-Scope` | Namespace for this request |
| `X-GistCache-Bypass` | `true` skips the cache entirely |
