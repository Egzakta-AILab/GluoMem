# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.6 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY gluomem ./gluomem
COPY artifacts ./artifacts

RUN uv sync --frozen --no-dev --extra postgres --extra service --extra llm

FROM python:3.12-slim AS runtime

RUN groupadd --system --gid 10001 gluomem \
    && useradd --system --uid 10001 --gid gluomem --create-home --home-dir /home/gluomem gluomem

WORKDIR /app

COPY --from=builder --chown=gluomem:gluomem /app/.venv /app/.venv
COPY --from=builder --chown=gluomem:gluomem /app/gluomem /app/gluomem
COPY --from=builder --chown=gluomem:gluomem /app/artifacts /app/artifacts

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER gluomem

EXPOSE 8080

ENTRYPOINT ["gluomem"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
