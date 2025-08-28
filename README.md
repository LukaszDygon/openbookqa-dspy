# openbookqa-dspy

DSPy-based agent for the OpenBookQA dataset using OpenAI gpt-4o. Managed with `uv`, Python 3.13, formatted by `black`, and type-checked by `mypy`.

## Prerequisites

- Python 3.13 available (uv can manage Python installs)
- `uv` installed: [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)
- OpenAI API key

## Quickstart
 
```bash
# 1) Install dependencies and create virtual env
uv sync

# 2) Provide environment variables (create .env or export)
cp .env.example .env
# then edit .env to include your OPENAI_API_KEY

# 3) Try the CLI
uv run openbookqa-dspy --help
```

## Environment

Create a `.env` file (local only; do not commit):
```env
OPENAI_API_KEY=sk-...
HF_HOME=~/.cache/huggingface
```

## Common Tasks

- Format: `uv run black .`
- Type check: `uv run mypy .`
- Run tests: `uv run pytest -q`

## Structure

- `src/openbookqa_dspy/agent.py` – DSPy model and optimization scaffolding
- `src/openbookqa_dspy/data.py` – Dataset loading utilities (Hugging Face `allenai/openbookqa`)
- `src/openbookqa_dspy/cli.py` – Typer CLI entry points

## Notes

- All functions are typed and include docstrings. Keep docstring style consistent.
- Initial agent is a placeholder; optimization strategies will be added iteratively.

