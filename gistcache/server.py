import time
import uuid

import httpx

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .embed import HashEmbedder
from .pipeline import Pipeline
from .providers import build_provider
from .store import Store


def build_pipeline(cfg: dict) -> Pipeline:
    store = Store(cfg["cache"]["path"], cfg["cache"]["max_entries"])
    try:
        from .embed import SentenceTransformerEmbedder
        from .verifier.cross import CrossEncoderScorer
        e = cfg.get("embedder", {})
        embedder = SentenceTransformerEmbedder(e.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
                                               e.get("device", "cpu"))
        scorer = CrossEncoderScorer(cfg.get("verifier", {}).get("cross_encoder", "cross-encoder/quora-distilroberta-base"))
    except ImportError:
        print("warning: install gistcache[ml] for the real embedder and cross-encoder; using hash fallback")
        embedder, scorer = HashEmbedder(), None
    small = build_provider(cfg["providers"]["small"]) if "small" in cfg["providers"] else None
    return Pipeline(cfg, store, embedder, build_provider(cfg["providers"]["large"]), small, scorer)


def create_app(pipeline: Pipeline) -> FastAPI:
    app = FastAPI(title="GistCache")

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "gistcache", "object": "model", "owned_by": "gistcache"}]}

    @app.post("/v1/chat/completions")
    async def chat(req: Request, x_gistcache_scope: str = Header("default"),
                   x_gistcache_bypass: str = Header("false")):
        body = await req.json()
        stream = body.pop("stream", False)
        body.pop("model", None)
        messages = body.pop("messages")
        has_tools = bool(body.pop("tools", None) or body.pop("functions", None))
        temperature = body.pop("temperature", None)
        try:
            r = pipeline.handle(messages, scope=x_gistcache_scope, bypass=x_gistcache_bypass.lower() == "true",
                                temperature=temperature, has_tools=has_tools, **body)
        except httpx.HTTPStatusError as e:
            return JSONResponse({"error": {"message": f"upstream provider returned {e.response.status_code}: "
                                                      f"{e.response.text[:300]}", "type": "upstream_error"}},
                                status_code=502)
        headers = {"X-GistCache-Tier": r.tier, "X-GistCache-Reason": r.reason,
                   "X-GistCache-Fallback": str(r.fallback).lower()}
        if r.source:
            headers["X-GistCache-Source"] = r.source
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
        if stream:
            import json

            def gen():
                for i in range(0, len(r.text), 40):
                    chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": "gistcache",
                             "choices": [{"index": 0, "delta": {"content": r.text[i:i + 40]}, "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                done = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": "gistcache",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                yield f"data: {json.dumps(done)}\n\ndata: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)
        return JSONResponse({"id": cid, "object": "chat.completion", "created": created, "model": "gistcache",
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": r.text},
                                          "finish_reason": "stop"}]}, headers=headers)

    return app
