from __future__ import annotations

"""MIPROv2 optimization candidate module builder."""

from typing import Callable, Optional
from pathlib import Path

import dspy

from .signature import Answerer


def _accuracy_metric() -> Callable[[dict[str, str], dspy.Prediction, dspy.Trace], float]:
    """Return a simple accuracy metric for MIPROv2 (1.0 if exact match else 0.0)."""

    def metric(
        example: dict[str, str],
        prediction: dspy.Prediction,
        trace: dspy.Trace,
    ) -> float:  # noqa: D401
        try:
            gold = str(example.get("answer", "")).strip().upper()
            pred = str(getattr(prediction, "answer", "")).strip().upper()
            return 1.0 if pred == gold else 0.0
        except Exception:
            return 0.0

    return metric


class MiproModule(dspy.Module):
    """MIPROv2-optimizable Chain-of-Thought program over `Answerer`."""

    def __init__(
        self,
        *,
        model_name: str,
        trainset: Optional[list[dict[str, str]]] = None,
        valset: Optional[list[dict[str, str]]] = None,
        max_iters: int = 3,
        seed: int = 13,
    ) -> None:
        super().__init__()
        program: dspy.Module = dspy.ChainOfThought(Answerer)

        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        model_safe = model_name.replace("/", "-").replace(":", "-").replace(" ", "_")
        save_path = models_dir / f"{model_safe}_miprov2.json"

        if save_path.exists():
            try:
                program = dspy.load(str(save_path))
            except Exception:
                # Fall through to (re)compile if possible
                pass

        # If not loaded and we have data, try to compile and save
        if (
            isinstance(program, dspy.Module)
            and program.__class__ is dspy.ChainOfThought
            and trainset
            and valset
        ):
            try:
                from dspy.optimizers import MIPROv2

                metric = _accuracy_metric()
                optimizer = MIPROv2(metric=metric, max_iters=max_iters, seed=seed)
                program = optimizer.compile(program, trainset=trainset, valset=valset)
                try:
                    dspy.save(program, str(save_path))
                except Exception:
                    # Non-fatal if saving fails
                    pass
            except Exception:
                # Optimizer not available; keep uncompiled program.
                pass
        self.program = program

    def forward(self, question: str, options: str) -> dspy.Prediction:
        return self.program(question=question, options=options)
