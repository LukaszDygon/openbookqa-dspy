from __future__ import annotations

from typing import Iterable, TypedDict, Optional, Literal
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


def prepare_examples(
    *, split: Literal["train", "validation", "test"], limit: Optional[int]
) -> list[QAExample]:
    """Load dataset split and return a (possibly truncated) list of `QAExample`.

    Args:
        split: Split name (train, validation, test).
        limit: Optional maximum number of examples.

    Returns:
        List of normalized QA examples.
    """
    ds = load_openbookqa()
    if split not in ds:
        raise RuntimeError(f"Split '{split}' not found. Available splits: {list(ds.keys())}")

    data_iter: Iterable[QAExample] = as_qa_iter(ds[split])
    if limit is not None:
        from itertools import islice

        data_iter = islice(data_iter, limit)
    return list(data_iter)


def format_choices(choices: list[str]) -> str:
    """Format a list of choices into "A. ...\nB. ..." string used by modules."""
    return "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(choices))


def to_mipro_examples(examples: list[QAExample]) -> list[dict[str, str]]:
    """Convert QA examples into dicts expected by MIPRO compile step."""
    return [
        {
            "question": ex["question"],
            "options": format_choices(ex["choices"]),
            "answer": ex["answer"],
        }
        for ex in examples
    ]
