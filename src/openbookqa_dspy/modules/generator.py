from __future__ import annotations

"""MIPROv2-optimized distractor generation module.

Generates three plausible but incorrect alternatives (distractors) given a
question and the correct answer text. The optimization metric rewards
sets that make a baseline Chain-of-Thought model answer correctly when
combined with the gold answer.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Callable, Iterable, Optional

import dspy
from dspy.teleprompt import MIPROv2

from ..config import Settings
from ..lm import configure_dspy_lm
from ..data import format_options
from .baseline import BaselineModule

logger = logging.getLogger(__name__)


class DistractorSignature(dspy.Signature):
    """Generate three distractors that are plausible but incorrect.

    Return exactly three options, each on its own line.
    """

    question: str = dspy.InputField()
    answer_text: str = dspy.InputField()
    distractors: str = dspy.OutputField(
        desc=(
            "Return exactly three plausible but incorrect options, one per line."
            " Do not include the correct answer."
        )
    )


def _parse_distractors(text: str, *, gold: str) -> list[str]:
    """Parse and sanitize distractors from a newline-separated string."""
    raw = [line.strip(" -•\t") for line in text.splitlines()]
    items: list[str] = [s for s in raw if s]
    # Deduplicate and filter out the gold answer
    uniq: list[str] = []
    seen: set[str] = set()
    gold_norm = gold.strip().lower()
    for s in items:
        s_norm = s.lower()
        if s_norm == gold_norm:
            continue
        if s_norm in seen:
            continue
        seen.add(s_norm)
        uniq.append(s)
    # Ensure exactly 3 elements (pad/truncate conservatively)
    if len(uniq) < 3:
        uniq = uniq + [f"Option {i}" for i in range(1, 4 - len(uniq) + 1)]
    return uniq[:3]


@dataclass(frozen=True)
class GeneratorBuildConfig:
    model_name: str
    seed: int = 13
    trainset: Optional[list[dspy.Example]] = None
    valset: Optional[list[dspy.Example]] = None


class MiproDistractorGenerator(dspy.Module):
    """MIPROv2-optimizable distractor generator program.

    Uses a metric that calls a baseline Chain-of-Thought answerer on the
    combined [gold + generated] options and rewards correctness.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        trainset: Optional[list[dspy.Example]] = None,
        valset: Optional[list[dspy.Example]] = None,
        seed: int = 13,
    ) -> None:
        super().__init__()
        # Ensure LM configured once
        configure_dspy_lm(settings)

        program: dspy.Module = dspy.ChainOfThought(DistractorSignature)

        # Prepare a baseline answerer and a metric that uses it
        baseline = BaselineModule()

        def metric(
            ex: dspy.Example, pred: dspy.Prediction, trace: dspy.Trace | None = None
        ) -> float:
            try:
                gold = ex.answer_text
                # Parse distractors
                gen = str(getattr(pred, "distractors", ""))
                ds = _parse_distractors(gen, gold=gold)
                options = [gold, *ds]
                formatted = format_options(options)
                ans = baseline(question=ex.question, options=formatted)
                picked = str(getattr(ans, "answer", "")).strip().upper()
                # correct is always index 0 => letter A
                return 1.0 if picked == "A" else 0.0
            except Exception:
                return 0.0

        # Load cached compiled program if available
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        model_safe = settings.model.replace("/", "-").replace(":", "-").replace(" ", "_")
        save_path = models_dir / f"{model_safe}_distractors_miprov2.json"

        if save_path.exists():
            logger.info("GEN: loading cached distractor program from %s", save_path)
            program = dspy.load(str(save_path))

        if (
            isinstance(program, dspy.Module)
            and program.__class__ is dspy.ChainOfThought
            and trainset
            and valset
        ):
            optimizer = MIPROv2(metric=metric, seed=seed, num_threads=16)
            logger.info(
                "GEN: compiling generator | model=%s | train=%d | val=%d | seed=%d",
                settings.model,
                len(trainset),
                len(valset),
                seed,
            )
            program = optimizer.compile(program, trainset=trainset, valset=valset)
            try:
                dspy.save(program, str(save_path))
                logger.info("GEN: saved compiled generator to %s", save_path)
            except Exception:
                logger.warning(
                    "GEN: failed to save compiled generator to %s (non-fatal)", save_path
                )

        self.program = program

    def forward(self, question: str, answer_text: str) -> dspy.Prediction:
        return self.program(question=question, answer_text=answer_text)

    def generate_distractors(self, question: str, answer_text: str) -> list[str]:
        """Convenience facade returning a clean list of 3 distractors."""
        pred = self.forward(question=question, answer_text=answer_text)
        text = str(getattr(pred, "distractors", ""))
        return _parse_distractors(text, gold=answer_text)
