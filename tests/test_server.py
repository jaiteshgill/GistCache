"""Integration tests for the OpenAI-compatible proxy, run against mock providers (no network, no models)."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from gistcache.config import load_config
from gistcache.embed import HashEmbedder
from gistcache.pipeline import Pipeline
from gistcache.providers import Completion
from gistcache.server import create_app
from gistcache.store import Store


class Fake:
    def __init__(self, model, reply):
        self.model, self.reply, self.calls = model, reply, 0

    def complete(self, messages, **params):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return Completion(self.reply, 10, 10)


def make_client(large_reply="L" * 80, small_reply="P" * 70):
    cfg = load_config(None)
    cfg["verifier"].update(hit_threshold=0.8, near_threshold=0.5)
    large, small = Fake("large", large_reply), Fake("small", small_reply)
    app = create_app(Pipeline(cfg, Store(), HashEmbedder(), large, small))
    return TestClient(app), large, small


def chat(client, content, headers=None, **body):
    return client.post("/v1/chat/completions", headers=headers or {},
                       json={"model": "gistcache", "messages": [{"role": "user", "content": content}], **body})


def test_models_endpoint():
    c, *_ = make_client()
    d = c.get("/v1/models").json()
    assert d["object"] == "list" and d["data"][0]["id"] == "gistcache"


def test_miss_then_hit_headers_and_body():
    c, large, _ = make_client()
    r1 = chat(c, "how do I reset my password")
    assert r1.status_code == 200
    assert r1.headers["x-gistcache-tier"] == "miss" and r1.headers["x-gistcache-fallback"] == "false"
    assert "x-gistcache-source" not in r1.headers
    body = r1.json()
    assert body["object"] == "chat.completion" and body["choices"][0]["message"]["content"] == "L" * 80
    r2 = chat(c, "how can I reset my password")
    assert r2.headers["x-gistcache-tier"] == "hit" and r2.headers["x-gistcache-reason"] == "hit: paraphrase"
    assert r2.headers["x-gistcache-source"] and large.calls == 1


def test_near_hit_header_and_patched_body():
    c, large, small = make_client()
    chat(c, "sort a list of numbers in python ascending")
    r = chat(c, "sort a list of numbers in python descending")
    assert r.headers["x-gistcache-tier"] == "near_hit" and r.headers["x-gistcache-reason"] == "near_hit: sort order"
    assert r.json()["choices"][0]["message"]["content"] == "P" * 70 and small.calls == 1


def test_cannot_patch_sets_fallback_header():
    c, large, _ = make_client(small_reply="CANNOT_PATCH")
    chat(c, "sort a list of numbers in python ascending")
    r = chat(c, "sort a list of numbers in python descending")
    assert r.headers["x-gistcache-tier"] == "miss" and r.headers["x-gistcache-fallback"] == "true"


def test_scope_header_isolates_tenants():
    c, large, _ = make_client()
    chat(c, "how do I reset my password", headers={"X-GistCache-Scope": "a"})
    assert chat(c, "how do I reset my password", headers={"X-GistCache-Scope": "b"}).headers["x-gistcache-tier"] == "miss"
    assert chat(c, "how do I reset my password", headers={"X-GistCache-Scope": "a"}).headers["x-gistcache-tier"] == "hit"


@pytest.mark.parametrize("kwargs,headers", [
    ({}, {"X-GistCache-Bypass": "true"}),
    ({"temperature": 1.2}, {}),
    ({"tools": [{"type": "function", "function": {"name": "f"}}]}, {}),
])
def test_bypass_paths_skip_the_cache(kwargs, headers):
    c, large, _ = make_client()
    r = chat(c, "hello there friend", headers=headers, **kwargs)
    assert r.headers["x-gistcache-tier"] == "bypass"
    chat(c, "hello there friend", headers=headers, **kwargs)
    assert large.calls == 2  # never cached, never served from cache


def test_streaming_reassembles_to_the_same_text_and_ends_with_done():
    c, *_ = make_client(large_reply="word " * 40)
    r = chat(c, "tell me something", stream=True)
    assert r.headers["content-type"].startswith("text/event-stream") and r.headers["x-gistcache-tier"] == "miss"
    lines = [l[6:] for l in r.text.splitlines() if l.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    chunks = [json.loads(l) for l in lines[:-1]]
    assert all(ch["object"] == "chat.completion.chunk" for ch in chunks)
    assert "".join(ch["choices"][0]["delta"].get("content", "") for ch in chunks) == "word " * 40
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_cached_answer_can_be_streamed():
    c, large, _ = make_client()
    chat(c, "how do I reset my password")
    r = chat(c, "how can I reset my password", stream=True)
    assert r.headers["x-gistcache-tier"] == "hit" and large.calls == 1


def test_upstream_error_becomes_502_with_the_providers_message():
    err = httpx.HTTPStatusError("bad", request=httpx.Request("POST", "http://x"),
                                response=httpx.Response(401, request=httpx.Request("POST", "http://x"), text="invalid x-api-key"))
    c, *_ = make_client(large_reply=err)
    r = chat(c, "anything at all")
    assert r.status_code == 502
    e = r.json()["error"]
    assert e["type"] == "upstream_error" and "401" in e["message"] and "invalid x-api-key" in e["message"]


def test_failed_request_is_not_cached():
    err = httpx.HTTPStatusError("bad", request=httpx.Request("POST", "http://x"),
                                response=httpx.Response(500, request=httpx.Request("POST", "http://x"), text="boom"))
    c, large, _ = make_client(large_reply=err)
    chat(c, "how do I reset my password")
    large.reply = "L" * 80
    assert chat(c, "how do I reset my password").headers["x-gistcache-tier"] == "miss"
