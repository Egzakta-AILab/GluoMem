.PHONY: install install-offline hooks test lint typecheck check wheel-check

PYTHON ?= 3.12

# Full development environment: all extras and dev tools.
install:
	uv sync --locked --extra all

# Offline checks: llm, service, postgres and dev tools, without model packages.
install-offline:
	uv sync --locked --extra llm --extra service --extra postgres

# One-time per clone: commit gate (hygiene + lint + typecheck) and push gate (tests)
hooks:
	uv run pre-commit install --hook-type pre-commit --hook-type pre-push

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run pyrefly check

check: lint typecheck test

# Build the wheel, install it core-only (pinned from uv.lock) into a fresh
# environment outside the checkout, and run the installed-wheel boundary check
# from that directory.
wheel-check:
	@set -eu; tmp=$$(mktemp -d /tmp/gluomem-wheel.XXXXXX); trap 'rm -rf "$$tmp"' EXIT; \
	uv build --wheel --out-dir "$$tmp/dist"; \
	uv export --locked --no-dev --no-emit-project --format requirements-txt -o "$$tmp/core.txt" > /dev/null; \
	uv venv --python $(PYTHON) "$$tmp/venv"; \
	uv pip install --python "$$tmp/venv/bin/python" -r "$$tmp/core.txt" "$$tmp"/dist/*.whl; \
	cd "$$tmp"; \
	"$$tmp/venv/bin/python" "$(CURDIR)/scripts/wheel_smoke.py" --expect-core-only; \
	"$$tmp/venv/bin/gluomem" --help > /dev/null
