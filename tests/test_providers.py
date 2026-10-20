import httpx
import pytest

from gistcache.providers import base
from gistcache.providers.base import Anthropic, OpenAICompatible

OK_ANTHROPIC = {"content": [{"type": "text", "text": "hello"}], "usage": {"input_tokens": 3, "output_tokens": 1}}


@pytest.fixture
def fake_http(monkeypatch):
    """Replace httpx.post with a script of responses; record sleeps instead of sleeping."""
    calls, sleeps = [], []
    script = []

    def post(url, **kw):
        calls.append((url, kw))
        item = script.pop(0)
        if isinstance(item, Exception):
            raise item
        status, body, headers = item
        return httpx.Response(status, json=body, headers=headers or {}, request=httpx.Request("POST", url))

    monkeypatch.setattr(base.httpx, "post", post)
    monkeypatch.setattr(base, "_sleep", sleeps.append)
    return script, calls, sleeps


def test_overloaded_and_rate_limited_requests_are_retried(fake_http):
    script, calls, sleeps = fake_http
    script += [(529, {"error": "overloaded"}, None), (429, {"error": "rate"}, {"retry-after": "3"}),
               (200, OK_ANTHROPIC, None)]
    c = Anthropic("m", "k").complete([{"role": "user", "content": "hi"}])
    assert c.text == "hello" and len(calls) == 3 and sleeps == [1, 3.0]  # backoff, then the server's retry-after


def test_network_errors_are_retried(fake_http):
    script, calls, sleeps = fake_http
    script += [httpx.ConnectError("reset"), httpx.ReadTimeout("slow"), (200, OK_ANTHROPIC, None)]
    assert Anthropic("m", "k").complete([{"role": "user", "content": "hi"}]).text == "hello"
    assert sleeps == [1, 2]


def test_permanent_errors_fail_immediately(fake_http):
    script, calls, sleeps = fake_http
    script += [(401, {"error": "bad key"}, None)]
    with pytest.raises(httpx.HTTPStatusError):
        Anthropic("m", "k").complete([{"role": "user", "content": "hi"}])
    assert len(calls) == 1 and sleeps == []


def test_gives_up_after_the_retry_budget(fake_http):
    script, calls, sleeps = fake_http
    script += [(529, {}, None)] * base.RETRIES
    with pytest.raises(httpx.HTTPStatusError):
        Anthropic("m", "k").complete([{"role": "user", "content": "hi"}])
    assert len(calls) == base.RETRIES and len(sleeps) == base.RETRIES - 1


def test_openai_compatible_retries_too(fake_http):
    script, calls, sleeps = fake_http
    script += [(503, {}, None), (200, {"choices": [{"message": {"content": "yo"}}], "usage": {}}, None)]
    assert OpenAICompatible("m", "http://x/v1").complete([{"role": "user", "content": "hi"}]).text == "yo"


def test_anthropic_request_shape(fake_http):
    script, calls, _ = fake_http
    script += [(200, OK_ANTHROPIC, None)]
    Anthropic("m", "k", workspace_id="ws").complete(
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}], max_tokens=7)
    url, kw = calls[0]
    assert url.endswith("/v1/messages") and kw["headers"]["anthropic-workspace-id"] == "ws"
    assert kw["json"]["system"] == "be brief" and kw["json"]["max_tokens"] == 7
    assert kw["json"]["messages"] == [{"role": "user", "content": "hi"}] and "temperature" not in kw["json"]
