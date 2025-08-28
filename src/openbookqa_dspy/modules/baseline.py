from __future__ import annotations

"""Baseline DSPy module builder (ChainOfThought over Answerer signature)."""

import dspy

from .signature import Answerer


class BaselineModule(dspy.Module):
    """Simple Chain-of-Thought over the `Answerer` signature."""

    def __init__(self) -> None:
        super().__init__()
        self.program = dspy.ChainOfThought(Answerer)

    def forward(self, question: str, options: str) -> dspy.Prediction:
        return self.program(question=question, options=options)
