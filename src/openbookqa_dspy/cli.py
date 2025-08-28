from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import typer

from .agent import build_pipeline
from .config import Settings
from .data import QAExample, as_qa_iter, load_openbookqa
from .eval import evaluate_with_details


app = typer.Typer(help="OpenBookQA DSPy Agent")


def _load_settings() -> Settings:
    """Load settings and ensure OPENAI_API_KEY is present."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        # Proceed if dotenv is not installed/available.
        pass
    settings = Settings.load()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    return settings


def _prepare_examples(split: str, limit: Optional[int]) -> list[QAExample]:
    """Load dataset split and return a (possibly truncated) list of examples."""
    ds = load_openbookqa()
    if split not in ds:
        raise RuntimeError(f"Split '{split}' not found. Available splits: {list(ds.keys())}")

    data_iter: Iterable[QAExample] = as_qa_iter(ds[split])
    if limit is not None:
        from itertools import islice

        data_iter = islice(data_iter, limit)
    return list(data_iter)


def _report_path_for(settings: Settings) -> Path:
    """Build a timestamped path under evaluation_results/ including the model name."""
    out_dir = Path("evaluation_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = settings.model.replace("/", "-").replace(":", "-").replace(" ", "_")
    return out_dir / f"{ts}_{model_safe}.json"


@app.command()
def eval(
    split: str = typer.Option("validation", help="Dataset split: train/validation/test"),
    limit: Optional[int] = typer.Option(50, help="Max examples to evaluate"),
) -> None:
    """Evaluate the baseline pipeline on OpenBookQA."""
    settings = _load_settings()
    examples = _prepare_examples(split=split, limit=limit)
    pipe = build_pipeline(settings)

    acc, n, records = evaluate_with_details(pipe, examples)
    report = {"sample_size": n, "accuracy": acc, "examples": records}
    out_path = _report_path_for(settings)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    typer.echo(f"Wrote JSON report to {out_path} (Accuracy@{split}: {acc:.3f}, n={n})")
