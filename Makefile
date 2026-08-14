.PHONY: install hooks test lint typecheck check optimize-prompts

install:
	uv sync

# One-time per clone: commit gate (hygiene + lint + typecheck) and push gate (tests)
hooks:
	uv run pre-commit install --hook-type pre-commit --hook-type pre-push

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run pyrefly check

check: lint typecheck test

# Offline DSPy compile -> artifacts/ (needs OPENROUTER_API_KEY + trainsets;
# spends real tokens). TRAINSETS defaults to benchmarks_data/trainsets.
optimize-prompts:
	uv run python -m gluomem.llm.optimize --task all --trainset $(or $(TRAINSETS),benchmarks_data/trainsets)
