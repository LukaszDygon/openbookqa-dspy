from __future__ import annotations

from typing import Iterable, TypedDict

import dspy
from .data import QAExample
from .agent import predict_answer


class ExampleResult(TypedDict):
    """Per-example evaluation result."""

    index: int
    question: str
    choices: list[str]
    expected: str
    predicted: str
    correct: bool


def evaluate(
    pipe: dspy.Module, examples: Iterable[QAExample]
) -> tuple[float, int, list[ExampleResult]]:
    """Evaluate and collect per-example details.

    Args:
        pipe: The DSPy pipeline to evaluate.
        examples: Iterable of `QAExample`.

    Returns:
        A tuple of (accuracy, sample_size, records).
    """
    records: list[ExampleResult] = []
    total = 0
    correct = 0
    for idx, ex in enumerate(examples):
        pred = predict_answer(pipe, ex)
        is_correct = pred.upper() == ex["answer"].upper()
        records.append(
            ExampleResult(
                index=idx,
                question=ex["question"],
                choices=ex["choices"],
                expected=ex["answer"],
                predicted=pred,
                correct=is_correct,
            )
        )
        total += 1
        if is_correct:
            correct += 1

    acc = 0.0 if total == 0 else correct / total
    return acc, total, records
