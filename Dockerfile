# ---- Stage 1: build ----
# Installs dependencies and downloads the verifier models. Nothing from this stage
# ships except the finished virtualenv and model cache.
FROM python:3.12-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# CPU-only PyTorch: the GPU build is ~2 GB larger and this service runs on CPU.
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy only what pip needs first, so code changes don't invalidate this cached layer.
COPY pyproject.toml README.md ./
COPY gistcache ./gistcache
RUN pip install ".[ml]"

# Bake the three verifier models into the image so the container starts without
# downloading anything and works even if Hugging Face is unreachable.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/quora-distilroberta-base'); \
CrossEncoder('cross-encoder/nli-deberta-v3-small')"

# ---- Stage 2: runtime ----
FROM python:3.12-slim

ENV PATH="/venv/bin:$PATH" \
    HF_HOME=/models \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1

# Run as a non-root user: if the app is ever compromised, the attacker isn't root in the container.
RUN useradd --create-home --uid 1000 gistcache
WORKDIR /app

COPY --from=build /venv /venv
COPY --from=build /models /models
COPY configs ./configs

# The SQLite cache lives here. Mount a volume on it so cached answers survive a redeploy.
RUN mkdir -p /app/data && chown -R gistcache:gistcache /app/data
VOLUME /app/data
USER gistcache

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/healthz', timeout=4)"

# Which provider config to use; override with -e GISTCACHE_CONFIG=configs/groq.yaml
ENV GISTCACHE_CONFIG=configs/anthropic.yaml

# Bind 0.0.0.0, not 127.0.0.1: inside a container, localhost is only reachable from inside the container.
# exec replaces the shell, so the app is PID 1 and receives Docker's stop signal directly.
CMD ["sh", "-c", "exec gistcache --config \"$GISTCACHE_CONFIG\" serve --host 0.0.0.0 --port 8100"]
