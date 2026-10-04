# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.13-slim-bookworm@sha256:5024f48ba9441d4b13a95d3945abc6365538e3a31109833367a1923523c6efed
FROM ${PYTHON_IMAGE} AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY pyproject.toml requirements.lock ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /opt/venv \
    && python -c "import tomllib; from pathlib import Path; Path('/tmp/runtime-requirements.txt').write_text('\n'.join(tomllib.loads(Path('pyproject.toml').read_text())['project']['dependencies']))" \
    && /opt/venv/bin/pip install --only-binary=:all: -c requirements.lock -r /tmp/runtime-requirements.txt
COPY README.md LICENSE ./
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install --no-deps -c requirements.lock . \
    && /opt/venv/bin/pip check

FROM ${PYTHON_IMAGE} AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    ORT_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1 \
    MODEL_CACHE_DIR=/app/.cache/embeddings
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /app/.cache/embeddings \
    && chown -R app:app /app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=12)"
CMD ["python", "-m", "uvicorn", "agentic_rag_assistant.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
