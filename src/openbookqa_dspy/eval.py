from __future__ import annotations

from typing import Iterable

import dspy
from .data import QAExample
from .agent import predict_answer


def accuracy(pipe: dspy.Module, examples: Iterable[QAExample]) -> float:
    """Compute simple accuracy over an iterable of QA examples.

    Args:
        pipe: The DSPy pipeline to evaluate.
        examples: Iterable of `QAExample`.

    Returns:
        Accuracy as a float in [0, 1].
    """
    total = 0
    correct = 0
    for ex in examples:
        total += 1
        pred = predict_answer(pipe, ex)
        if pred[:1].upper() == ex["answer"][:1].upper():
            correct += 1
    return 0.0 if total == 0 else correct / total
