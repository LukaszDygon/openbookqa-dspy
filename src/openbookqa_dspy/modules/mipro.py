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

        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        model_safe = model_name.replace("/", "-").replace(":", "-").replace(" ", "_")
        save_path = models_dir / f"{model_safe}_miprov2.json"

        if save_path.exists():
            logger.info("MIPRO: loading cached program from %s", save_path)
            program = dspy.load(str(save_path))

        # If not loaded and we have data, try to compile and save
        if (
            isinstance(program, dspy.Module)
            and program.__class__ is dspy.ChainOfThought
            and trainset
            and valset
        ):

            optimizer = MIPROv2(metric=metric, seed=seed, num_threads=16)
            logger.info(
                "MIPRO: starting compile | model=%s | train=%d | val=%d | seed=%d",
                model_name,
                len(trainset),
                len(valset),
                seed,
            )
            program = optimizer.compile(program, trainset=trainset, valset=valset)
            try:
                dspy.save(program, str(save_path))
                logger.info("MIPRO: saved compiled program to %s", save_path)
            except Exception:
                # Non-fatal if saving fails
                logger.warning(
                    "MIPRO: failed to save compiled program to %s (non-fatal)", save_path
                )
        self.program = program

    def forward(self, question: str, options: str) -> dspy.Prediction:
        return self.program(question=question, options=options)
