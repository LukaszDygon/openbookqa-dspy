from __future__ import annotations

import logging
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
from .data import prepare_q_with_answer_examples
from .modules.generator import MiproDistractorGenerator


app = typer.Typer(help="OpenBookQA DSPy Agent")


def _load_settings() -> Settings:
    """Load settings and ensure OPENAI_API_KEY is present."""
    load_dotenv()
    settings = Settings.load()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    return settings


def _configure_logging() -> None:
    """Initialize basic logging format once for CLI sessions."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _report_path_for(settings: Settings, approach: ApproachEnum, n_eval: int) -> Path:
    """Build a timestamped path under evaluation_results/ including approach and size.

    The filename format is: {timestamp}_{approach}-{n_eval}.json
    """
    out_dir = Path("evaluation_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    approach_safe = approach.value.replace("/", "-").replace(":", "-").replace(" ", "_")
    return out_dir / f"{ts}_{approach_safe}-{n_eval}.json"


@app.command()
def eval(
    limit: Optional[int] = typer.Option(50, help="Max examples to evaluate"),
    approach: ApproachEnum = typer.Option(
        ApproachEnum.baseline, help="Which approach to evaluate: baseline or mipro"
    ),
    train_limit: Optional[int] = typer.Option(
        None, help="(mipro) Number of training examples to compile with"
    ),
    val_limit: Optional[int] = typer.Option(
        None, help="(mipro) Number of validation examples for compilation tuning"
    ),
    seed: int = typer.Option(13, help="(mipro) Random seed for optimization"),
    threads: int = typer.Option(16, help="Evaluation threads (1 = serial)"),
) -> None:
    """Evaluate a selected approach on OpenBookQA (baseline or MIPROv2).

    Always evaluates on the validation split; optimization uses train/validation as needed.
    """
    _configure_logging()
    log = logging.getLogger(__name__)
    settings = _load_settings()
    examples = prepare_examples(split="validation", limit=limit)
    n_val = len(examples)
    log.info(
        "Eval start | model=%s | approach=%s | limit=%s | train_limit=%s | val_limit=%s | seed=%d | threads=%d | n_val=%s",
        settings.model,
        approach.value,
        limit,
        train_limit,
        val_limit,
        seed,
        threads,
        n_val if n_val >= 0 else "unknown",
    )
    pipe = build_selected_pipeline(
        settings,
        approach=approach,
        train_limit=train_limit,
        val_limit=val_limit,
        seed=seed,
    )

    acc, n, records = evaluate(pipe, examples, threads=threads)
    report = {"sample_size": n, "accuracy": acc, "examples": records}
    out_path = _report_path_for(settings, approach, n)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    log.info("Eval done | wrote=%s | accuracy=%.3f | n=%d", out_path, acc, n)
    typer.echo(
        f"Wrote JSON report to {out_path} (Approach={approach}, Accuracy@validation: {acc:.3f}, n={n})"
    )


@app.command()
def generate_questions(
    limit: Optional[int] = typer.Option(None, help="Max examples from test split (None = all)"),
    train_limit: Optional[int] = typer.Option(
        None, help="(mipro) Number of training examples for generator compilation"
    ),
    val_limit: Optional[int] = typer.Option(
        None, help="(mipro) Number of validation examples for generator compilation"
    ),
    seed: int = typer.Option(13, help="(mipro) Random seed for optimization"),
    output: Path = typer.Option(Path("questions.json"), help="Output JSON file path"),
) -> None:
    """Generate multiple-choice questions with 3 optimized distractors.

    Uses the test split of OpenBookQA. The correct answer is placed as option A
    and three generated distractors fill B-D. The file is written as a list of
    objects with fields: question, options (list[str] of length 4), answerKey ("A").
    """
    _configure_logging()
    log = logging.getLogger(__name__)
    settings = _load_settings()

    # Prepare datasets for generator training/validation and final test examples
    trainset = (
        prepare_q_with_answer_examples(split="train", limit=train_limit) if train_limit else None
    )
    valset = (
        prepare_q_with_answer_examples(split="validation", limit=val_limit) if val_limit else None
    )
    test_examples = prepare_q_with_answer_examples(split="test", limit=limit)
    n_test = len(test_examples)

    log.info(
        "Gen start | model=%s | train_limit=%s | val_limit=%s | seed=%d | n_test=%d",
        settings.model,
        train_limit,
        val_limit,
        seed,
        n_test,
    )

    generator = MiproDistractorGenerator(settings, trainset=trainset, valset=valset, seed=seed)

    records: list[dict[str, object]] = []
    for idx, ex in enumerate(test_examples):
        ds = generator.generate_distractors(ex.question, ex.answer_text)
        options = [ex.answer_text, *ds]
        records.append(
            {
                "index": idx,
                "question": ex.question,
                "options": options,
                "answerKey": "A",
            }
        )

    output.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    log.info("Generation done | wrote=%s | n=%d", output, len(records))
    typer.echo(f"Wrote questions to {output} (n={len(records)})")
