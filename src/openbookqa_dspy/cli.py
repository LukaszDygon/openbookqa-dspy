from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
import typer
from typing import Optional
from .config import Settings
from .data import load_openbookqa, as_qa_iter
from .agent import build_pipeline
from .eval import evaluate

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

    # Materialize the (possibly truncated) iterator to reuse for both eval and report
    examples = list(data_iter)

    pipe = build_pipeline(settings)

    acc, n, records = evaluate(pipe, examples)
    report = {"sample_size": n, "accuracy": acc, "examples": records}

    # Prepare output directory and timestamped filename with model name
    out_dir = Path("evaluation_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = settings.model.replace("/", "-").replace(":", "-").replace(" ", "_")
    out_path = out_dir / f"{ts}_{model_safe}.json"

    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    typer.echo(f"Wrote JSON report to {out_path} (Accuracy@{split}: {acc:.3f}, n={n})")
