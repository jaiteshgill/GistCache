import os
from dataclasses import dataclass
from typing import Protocol

import httpx


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
        r = httpx.post(f"{self.base_url}/chat/completions", headers=headers, timeout=120,
                       json={"model": self.model, "messages": messages, **params})
        r.raise_for_status()
        d = r.json()
        u = d.get("usage", {})
        return Completion(d["choices"][0]["message"]["content"] or "",
                          u.get("prompt_tokens", 0), u.get("completion_tokens", 0))


class Anthropic:
    def __init__(self, model: str, api_key: str, base_url: str = "https://api.anthropic.com"):
        self.model, self.api_key, self.base_url = model, api_key, base_url.rstrip("/")

    def complete(self, messages, **params) -> Completion:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        msgs = [m for m in messages if m["role"] != "system"]
        body = {"model": self.model, "messages": msgs, "max_tokens": params.get("max_tokens", 1024)}
        if system:
            body["system"] = system
        if "temperature" in params:
            body["temperature"] = params["temperature"]
        r = httpx.post(f"{self.base_url}/v1/messages", timeout=120, json=body,
                       headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
        r.raise_for_status()
        d = r.json()
        return Completion("".join(b.get("text", "") for b in d["content"]),
                          d["usage"]["input_tokens"], d["usage"]["output_tokens"])


def build_provider(cfg: dict) -> Provider:
    kind = cfg["kind"]
    key = os.environ.get(cfg.get("api_key_env", ""), "")
    if kind == "anthropic":
        return Anthropic(cfg["model"], key, cfg.get("base_url", "https://api.anthropic.com"))
    default = {"openai": "https://api.openai.com/v1", "ollama": "http://localhost:11434/v1"}.get(kind, "")
    return OpenAICompatible(cfg["model"], cfg.get("base_url", default), key)
