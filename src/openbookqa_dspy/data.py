from __future__ import annotations

from typing import Iterable, TypedDict
from datasets import Dataset, DatasetDict, load_dataset


class QAExample(TypedDict):
    """Single QA item.

    Attributes:
        question: Question text.
        choices: List of answer choices.
        answer: The correct choice label (e.g., "A", "B", "C", or "D").
    """

    question: str
    choices: list[str]
    answer: str


def load_openbookqa() -> DatasetDict:
    """Load the OpenBookQA dataset from Hugging Face.

    Returns:
        A `DatasetDict` with splits like train/dev/test.
    """
    ds = load_dataset("allenai/openbookqa")
    return ds


def as_qa_iter(dataset: Dataset) -> Iterable[QAExample]:
    """Map a HF `Dataset` to an iterator of normalized `QAExample`.

    Args:
        dataset: A Hugging Face `Dataset` split.

    Yields:
        Normalized `QAExample` items.
    """
    for row in dataset:
        yield QAExample(
            question=row.get("question_stem", ""),
            choices=row.get("choices", {}).get("text", []),
            answer=row.get("answerKey", ""),
        )
