---
trigger: always_on
---

# Windsurf Agent Rules

These rules guide how Cascade should operate on this project.

 

- Project: `openbookqa-dspy`
- Conversation Anchor: @conversation:"Windsurf Agent Rules Setup" (Cascade ID: 276c1037-98a1-4c9e-b4be-c6be8b06641c)
- Tech stack: Python 3.13, uv (package manager), DSPy, OpenAI API (gpt-4o), Hugging Face Datasets (OpenBookQA), mypy, black

 

## Engineering Standards

 

- Use Python 3.13. All functions must be fully typed and have consistent, concise docstrings (Google style or NumPy style acceptable; be consistent).
- Formatting: `black` with line-length 100 and target-version py313.
- Type checking: `mypy` in strict-ish mode (`disallow_untyped_defs = true`, etc.).
- Prefer pure functions and small modules. Follow clean code principles: clear naming, single responsibility, no redundant comments, avoid deep nesting.
- Add logging where helpful; avoid blanket exceptions.

 

## Project Conventions

 

- Package name: `openbookqa_dspy` under `src/` layout.
- Environment via `uv`. Dependencies in `pyproject.toml` using PEP 621 and `[tool.uv]` dev-dependencies.
- Secrets via environment variables. Use `.env` locally (not committed).
- Dataset: `allenai/openbookqa` from Hugging Face.
- Model: OpenAI `gpt-4o` via the official `openai` Python SDK.

 

## Agent Guidelines

 

- Build the agent using `dspy-ai`. Start with a simple pipeline, then add optimization via DSPy (e.g., `BootstrapFewShot`, `COPRO`, or `MIPRO` strategies) as appropriate.
- Keep dataset handling isolated in `data.py`; model and DSPy logic in `agent.py`; CLI in `cli.py`.
- Ensure deterministic evaluation utilities for repeatability (fixed seeds where meaningful).

 

## Tooling Commands (reference)

 

- Setup: `uv sync`
- Run: `uv run openbookqa-dspy --help`
- Format: `uv run black .`
- Type check: `uv run mypy .`
- Tests: `uv run pytest -q`

 

## Documentation

 

- Keep `README.md` updated with run instructions and environment variables.
- Document any new modules and public functions with docstrings.
