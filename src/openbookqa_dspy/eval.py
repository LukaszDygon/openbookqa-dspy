from __future__ import annotations

import logging
from typing import Iterable, TypedDict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

import dspy
from .data import QAExample
from .agent import predict_answer


logger = logging.getLogger(__name__)


class ExampleResult(TypedDict):
    """Per-example evaluation result."""

    index: int
    question: str
    choices: list[str]
    expected: str
    predicted: str
    correct: bool


def evaluate(
    pipe: dspy.Module, examples: Iterable[QAExample], *, threads: int = 1
) -> tuple[float, int, list[ExampleResult]]:
    """Evaluate and collect per-example details.

    Args:
        pipe: The DSPy pipeline to evaluate.
        examples: Iterable of `QAExample`.
        threads: Number of threads to use (1 = serial).

    Returns:
        A tuple of (accuracy, sample_size, records).
    """
    if threads <= 1:
        serial_records: list[ExampleResult] = []
        total = 0
        correct = 0
        logger.info("Starting evaluation over examples (size unknown a priori)")
        for idx, ex in enumerate(examples):
            pred = predict_answer(pipe, ex)
            is_correct = pred.upper() == ex["answer"].upper()
            serial_records.append(
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
            # Log lightweight progress every 10 examples to avoid excessive verbosity.
            if total % 10 == 0:
                running_acc = 0.0 if total == 0 else correct / total
                logger.info(
                    "Progress: processed %d examples; running accuracy=%.3f", total, running_acc
                )

        acc = 0.0 if total == 0 else correct / total
        logger.info("Finished evaluation: n=%d, accuracy=%.3f", total, acc)
        return acc, total, serial_records

    # Parallel path: realize examples into a list for indexing and to avoid double iteration
    ex_list: list[QAExample] = list(examples)
    n_total = len(ex_list)
    logger.info("Starting parallel evaluation with %d threads over %d examples", threads, n_total)

    def _task(idx_ex: Tuple[int, QAExample]) -> Tuple[int, str]:
        idx, ex = idx_ex
        pred = predict_answer(pipe, ex)
        return idx, pred

    futures: list[Future[Tuple[int, str]]] = []
    par_records: list[ExampleResult] = []
    correct = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=threads) as tp:
        for idx, ex in enumerate(ex_list):
            futures.append(tp.submit(_task, (idx, ex)))

        for fut in as_completed(futures):
            idx, pred = fut.result()
            ex = ex_list[idx]
            is_correct = pred.upper() == ex["answer"].upper()
            par_records.append(
                ExampleResult(
                    index=idx,
                    question=ex["question"],
                    choices=ex["choices"],
                    expected=ex["answer"],
                    predicted=pred,
                    correct=is_correct,
                )
            )
            if is_correct:
                correct += 1
            completed += 1
            if completed % 10 == 0:
                running_acc = 0.0 if completed == 0 else correct / completed
                logger.info(
                    "Progress: processed %d/%d examples; running accuracy=%.3f",
                    completed,
                    n_total,
                    running_acc,
                )

    # Finalize
    acc = 0.0 if n_total == 0 else correct / n_total
    logger.info("Finished evaluation: n=%d, accuracy=%.3f", n_total, acc)
    par_records.sort(key=lambda r: r["index"])
    return acc, n_total, par_records
