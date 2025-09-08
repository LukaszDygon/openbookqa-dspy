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

## Run the Web UI

The project ships with a small Flask web UI to browse evaluation results and review questions.

```bash
# 1) Install deps (if not already)
uv sync

# 2) Ensure environment variables are set (see Environment section)

# 3) Start the web server (debug mode with auto-reload)
uv run openbookqa-dspy-web
```

Then open:

- Evaluations (answers): [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
- Questions review: [http://127.0.0.1:5000/questions](http://127.0.0.1:5000/questions)

Notes:

- The server binds to `127.0.0.1:5000` by default.
- Debug mode is enabled for local development with hot reload.
- If you see `404` on `/questions`, make sure you are on the latest code and restart the server so routes are registered.

## Structure

- `src/openbookqa_dspy/agent.py` – DSPy model and optimization scaffolding
- `src/openbookqa_dspy/data.py` – Dataset loading utilities (Hugging Face `allenai/openbookqa`)
- `src/openbookqa_dspy/cli.py` – Typer CLI entry points

## Notes

- All functions are typed and include docstrings. Keep docstring style consistent.
- Initial agent is a placeholder; optimization strategies will be added iteratively.

