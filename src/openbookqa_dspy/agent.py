from __future__ import annotations

import dspy
from .config import Settings
from .data import QAExample, format_choices, prepare_examples, to_mipro_examples
from .lm import configure_dspy_lm
from .modules.baseline import BaselineModule
from .modules.mipro import MiproModule
from .modules import ApproachEnum


def _configure_dspy_lm(settings: Settings) -> None:
    """Backwards-compatible wrapper around `configure_dspy_lm()`.

    Kept to avoid touching other call sites; delegates to shared util.
    """
    configure_dspy_lm(settings)


def build_pipeline(settings: Settings) -> dspy.Module:
    """Create a minimal DSPy pipeline using OpenAI gpt-4o.

    Args:
        settings: Loaded runtime settings.

    Returns:
        A `dspy.Module` pipeline (baseline).
    """
    _configure_dspy_lm(settings)
    # Delegate to modules.baseline for the default behavior.
    return BaselineModule()


def predict_answer(pipe: dspy.Module, example: QAExample) -> str:
    """Run the pipeline on a single example.

    Args:
        pipe: The DSPy pipeline.
        example: An input example with question/choices/answer.

    Returns:
        Predicted answer letter.
    """
    opts = format_choices(example["choices"])
    pred = pipe(question=example["question"], options=opts)
    return str(getattr(pred, "answer", "")).strip()


def build_selected_pipeline(
    settings: Settings,
    *,
    approach: ApproachEnum,
    train_limit: int | None,
    val_limit: int | None,
    max_iters: int,
    seed: int,
) -> dspy.Module:
    """Build a pipeline for the chosen approach.

    - baseline: simple ChainOfThought, no optimization
    - mipro: optional compilation using train/validation splits
    """
    configure_dspy_lm(settings)
    if approach == ApproachEnum.baseline:
        return BaselineModule()

    if approach == ApproachEnum.mipro:
        trainset = valset = None
        if train_limit is not None and train_limit > 0:
            train_examples = prepare_examples(split="train", limit=train_limit)
            trainset = to_mipro_examples(train_examples)
        if val_limit is not None and val_limit > 0:
            val_examples = prepare_examples(split="validation", limit=val_limit)
            valset = to_mipro_examples(val_examples)
        return MiproModule(
            model_name=settings.model,
            trainset=trainset,
            valset=valset,
            max_iters=max_iters,
            seed=seed,
        )

    # Should not happen due to typing; make explicit if it does.
    raise ValueError(f"Unknown approach: {approach.value}")
