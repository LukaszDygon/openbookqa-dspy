from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

import typer

from .agent import build_selected_pipeline
from .config import Settings
from .data import prepare_examples
from .eval import evaluate
from .modules import ApproachEnum


app = typer.Typer(help="OpenBookQA DSPy Agent")


def _load_settings() -> Settings:
    """Load settings and ensure OPENAI_API_KEY is present."""
    load_dotenv()
    settings = Settings.load()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    return settings


def _report_path_for(settings: Settings) -> Path:
    """Build a timestamped path under evaluation_results/ including the model name."""
    out_dir = Path("evaluation_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = settings.model.replace("/", "-").replace(":", "-").replace(" ", "_")
    return out_dir / f"{ts}_{model_safe}.json"


@app.command()
def eval(
    limit: Optional[int] = typer.Option(50, help="Max examples to evaluate"),
    approach: ApproachEnum = typer.Option(
        ApproachEnum.baseline,
        help="Which approach to evaluate: baseline or mipro"
    ),
    train_limit: Optional[int] = typer.Option(
        None, help="(mipro) Number of training examples to compile with"
    ),
    val_limit: Optional[int] = typer.Option(
        None, help="(mipro) Number of validation examples for compilation tuning"
    ),
    max_iters: int = typer.Option(3, help="(mipro) Max optimization iterations"),
    seed: int = typer.Option(13, help="(mipro) Random seed for optimization"),
) -> None:
    """Evaluate a selected approach on OpenBookQA (baseline or MIPROv2).

    Always evaluates on the validation split; optimization uses train/validation as needed.
    """
    settings = _load_settings()
    examples = prepare_examples(split="validation", limit=limit)
    pipe = build_selected_pipeline(
        settings,
        approach=approach,
        train_limit=train_limit,
        val_limit=val_limit,
        max_iters=max_iters,
        seed=seed,
    )

    acc, n, records = evaluate(pipe, examples)
    report = {"sample_size": n, "accuracy": acc, "examples": records}
    out_path = _report_path_for(settings)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    typer.echo(
        f"Wrote JSON report to {out_path} (Approach={approach}, Accuracy@validation: {acc:.3f}, n={n})"
    )
