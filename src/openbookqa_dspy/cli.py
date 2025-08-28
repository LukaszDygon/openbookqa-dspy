from __future__ import annotations

import typer
from typing import Optional
from .config import Settings
from .data import load_openbookqa, as_qa_iter
from .agent import build_pipeline
from .eval import accuracy

app = typer.Typer(help="OpenBookQA DSPy Agent")


@app.command()
def eval(
    split: str = typer.Option("validation", help="Dataset split: train/validation/test"),
    limit: Optional[int] = typer.Option(50, help="Max examples to evaluate"),
) -> None:
    """Evaluate the baseline pipeline on OpenBookQA."""
    # Load environment variables from a local .env if present.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        # Proceed if dotenv is not installed/available.
        pass
    settings = Settings.load()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    ds = load_openbookqa()
    if split not in ds:
        raise RuntimeError(f"Split '{split}' not found. Available splits: {list(ds.keys())}")

    data_iter = as_qa_iter(ds[split])
    if limit is not None:
        from itertools import islice

        data_iter = islice(data_iter, limit)

    pipe = build_pipeline(settings)
    acc = accuracy(pipe, data_iter)
    typer.echo(f"Accuracy@{split} (n={limit or 'all'}): {acc:.3f}")
