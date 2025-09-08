from __future__ import annotations

"""MIPROv2 optimization candidate module builder."""

import logging
from typing import Optional
from pathlib import Path
from dspy.teleprompt import MIPROv2

import dspy

from .signature import Answerer

logger = logging.getLogger(__name__)


def metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: dspy.Trace = None,
) -> float:  # noqa: D401
    try:
        gold = example.answer.upper()
        pred = str(getattr(prediction, "answer", "")).strip().upper()
        return 1.0 if pred == gold else 0.0
    except Exception:
        return 0.0


def _safe_model_name(name: str) -> str:
    """Return a filesystem-safe model name.

    Replaces path separators, colons, and spaces with safe characters for filenames.
    """
    return name.replace("/", "-").replace(":", "-").replace(" ", "_")


def _save_path_for(model_name: str) -> Path:
    """Compute the JSON state file path under `models/` for the given model name."""
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_safe = _safe_model_name(model_name)
    return models_dir / f"{model_safe}_miprov2.json"


def _load_state_if_exists(program: dspy.Module, save_path: Path) -> None:
    """Load state-only JSON into the given `program` if the path exists."""
    if save_path.exists():
        logger.info("MIPRO: loading cached program from %s", save_path)
        program.load(str(save_path))


def _save_state(program: dspy.Module, save_path: Path) -> None:
    """Persist program state-only JSON; logs errors but does not raise."""
    try:
        program.save(str(save_path), save_program=False)
        logger.info("MIPRO: saved compiled program to %s", save_path)
    except Exception:
        logger.warning(
            "MIPRO: failed to save compiled program to %s (non-fatal)", save_path
        )


def _compile_with_mipro(
    program: dspy.Module,
    *,
    trainset: list[dspy.Example],
    valset: list[dspy.Example],
    seed: int,
    model_name: str
) -> dspy.Module:
    """Compile the program with MIPROv2 and persist its state to JSON.

    Returns the compiled program.
    """
    optimizer = MIPROv2(metric=metric, seed=seed, num_threads=16, log_dir='logs')
    logger.info(
        "MIPRO: starting compile | model=%s | train=%d | val=%d | seed=%d",
        model_name,
        len(trainset),
        len(valset),
        seed,
    )
    compiled = optimizer.compile(program, trainset=trainset, valset=valset)

    return compiled


class MiproModule(dspy.Module):
    """MIPROv2-optimizable Chain-of-Thought program over `Answerer`."""

    def __init__(
        self,
        *,
        model_name: str,
        trainset: Optional[list[dspy.Example]] = None,
        valset: Optional[list[dspy.Example]] = None,
        seed: int = 13,
    ) -> None:
        super().__init__()
        program: dspy.Module = dspy.ChainOfThought(Answerer)

        save_path = _save_path_for(model_name)
        _load_state_if_exists(program, save_path)

        if (
            isinstance(program, dspy.Module)
            and program.__class__ is dspy.ChainOfThought
            and trainset
            and valset
        ):
            program = _compile_with_mipro(
                program,
                trainset=trainset,
                valset=valset,
                seed=seed,
                model_name=model_name
            )

            _save_state(program, save_path)

            self.program = program

    def forward(self, question: str, options: str) -> dspy.Prediction:
        return self.program(question=question, options=options)
