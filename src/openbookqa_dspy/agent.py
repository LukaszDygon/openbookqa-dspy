from __future__ import annotations

import os
import dspy
from .config import Settings
from .data import QAExample


def _configure_dspy_lm(settings: Settings) -> None:
    """Configure DSPy to use its built-in LM with OpenAI provider.

    Sets the OPENAI_API_KEY if provided in settings and initializes a deterministic
    LM suitable for short JSON-adapted completions.
    """
    if settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key

    lm = dspy.LM(
        model=settings.model,
        cache=False,
    )
    dspy.settings.configure(lm=lm)


def build_pipeline(settings: Settings) -> dspy.Module:
    """Create a minimal DSPy pipeline using OpenAI gpt-4o.

    Args:
        settings: Loaded runtime settings.

    Returns:
        A `dspy.Module` pipeline (baseline).
    """
    _configure_dspy_lm(settings)

    class Answerer(dspy.Signature):
        """Answer the multiple-choice question by returning the single best option (A-D)."""

        question: str = dspy.InputField()
        options: str = dspy.InputField()
        answer: str = dspy.OutputField(desc="Only the single letter A, B, C, or D.")

    return dspy.ChainOfThought(Answerer)  # simple baseline; replace with optimizer later


def predict_answer(pipe: dspy.Module, example: QAExample) -> str:
    """Run the pipeline on a single example.

    Args:
        pipe: The DSPy pipeline.
        example: An input example with question/choices/answer.

    Returns:
        Predicted answer letter.
    """
    opts = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(example["choices"]))
    pred = pipe(question=example["question"], options=opts)
    return str(getattr(pred, "answer", "")).strip()
