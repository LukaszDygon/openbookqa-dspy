from __future__ import annotations

from typing import Iterable, Optional, Literal
from datasets import Dataset, DatasetDict, load_dataset
import dspy

def format_options(options: list[str]) -> list[str]:
    return [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]

def load_openbookqa() -> DatasetDict:
    """Load the OpenBookQA dataset from Hugging Face.

    Returns:
        A `DatasetDict` with splits like train/dev/test.
    """
    ds = load_dataset("allenai/openbookqa")
    return ds


def as_qa_iter(dataset: Dataset) -> Iterable[dspy.Example]:
    """Map a HF `Dataset` to an iterator of normalized `dspy.Example`.

    Args:
        dataset: A Hugging Face `Dataset` split.

    Yields:
        Normalized `dspy.Example` items.
    """
    for row in dataset:
        yield dspy.Example(
            question=row.get("question_stem", ""),
            options=format_options(row.get("choices", {}).get("text", [])),
            answer=row.get("answerKey", ""),
        ).with_inputs("question", "options")


def prepare_examples(
    *, split: Literal["train", "validation", "test"], limit: Optional[int]
) -> list[dspy.Example]:
    """Load dataset split and return a (possibly truncated) list of `dspy.Example`.

    Args:
        split: Split name (train, validation, test).
        limit: Optional maximum number of examples.

    Returns:
        List of normalized QA examples.
    """
    ds = load_openbookqa()
    if split not in ds:
        raise RuntimeError(f"Split '{split}' not found. Available splits: {list(ds.keys())}")

    data_iter: Iterable[dspy.Example] = as_qa_iter(ds[split])
    if limit is not None:
        from itertools import islice

        data_iter = islice(data_iter, limit)
    return list(data_iter)
