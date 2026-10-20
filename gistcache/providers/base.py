import os
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

# Transient failures worth retrying: rate limits, overload (Anthropic 529), gateway errors, timeouts.
# Anything else (400 bad request, 401 bad key, 404) fails immediately because retrying cannot fix it.
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
RETRIES = 5
_sleep = time.sleep  # swapped out in tests


def _post(url: str, **kw) -> httpx.Response:
    for attempt in range(RETRIES):
        last = attempt == RETRIES - 1
        try:
            r = httpx.post(url, **kw)
        except httpx.TransportError:  # connection reset, DNS hiccup, read timeout
            if last:
                raise
            _sleep(2 ** attempt)
            continue
        if r.status_code in RETRY_STATUS and not last:
            try:
                wait = float(r.headers.get("retry-after", 2 ** attempt))
            except ValueError:
                wait = 2 ** attempt
            _sleep(min(wait, 30))
            continue
        r.raise_for_status()
        return r
    raise AssertionError("unreachable")


@dataclass
class Completion:
    text: str
    in_tokens: int = 0
    out_tokens: int = 0


class Provider(Protocol):
    model: str

    def complete(self, messages: list[dict], **params) -> Completion: ...


class OpenAICompatible:
    """Works for OpenAI, Ollama (/v1) and any OpenAI-compatible server."""

    def __init__(self, model: str, base_url: str, api_key: str = ""):
        self.model, self.base_url, self.api_key = model, base_url.rstrip("/"), api_key

    def complete(self, messages, **params) -> Completion:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = _post(f"{self.base_url}/chat/completions", headers=headers, timeout=120,
                  json={"model": self.model, "messages": messages, **params})
        d = r.json()
        u = d.get("usage", {})
        return Completion(d["choices"][0]["message"]["content"] or "",
                          u.get("prompt_tokens", 0), u.get("completion_tokens", 0))


class Anthropic:
    def __init__(self, model: str, api_key: str, base_url: str = "https://api.anthropic.com",
                 workspace_id: str = ""):
        self.model, self.api_key, self.base_url = model, api_key, base_url.rstrip("/")
        self.workspace_id = workspace_id

    def complete(self, messages, **params) -> Completion:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        msgs = [m for m in messages if m["role"] != "system"]
        body = {"model": self.model, "messages": msgs, "max_tokens": params.get("max_tokens", 1024)}
        if system:
            body["system"] = system
        if "temperature" in params:
            body["temperature"] = params["temperature"]
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        if self.workspace_id:
            headers["anthropic-workspace-id"] = self.workspace_id
        r = _post(f"{self.base_url}/v1/messages", timeout=120, json=body, headers=headers)
        d = r.json()
        return Completion("".join(b.get("text", "") for b in d["content"]),
                          d["usage"]["input_tokens"], d["usage"]["output_tokens"])


def build_provider(cfg: dict) -> Provider:
    kind = cfg["kind"]
    key = os.environ.get(cfg.get("api_key_env", ""), "")
    if kind == "anthropic":
        ws = os.environ.get(cfg.get("workspace_id_env", "ANTHROPIC_WORKSPACE_ID"), "")
        return Anthropic(cfg["model"], key, cfg.get("base_url", "https://api.anthropic.com"), ws)
    default = {"openai": "https://api.openai.com/v1", "ollama": "http://localhost:11434/v1"}.get(kind, "")
    return OpenAICompatible(cfg["model"], cfg.get("base_url", default), key)
