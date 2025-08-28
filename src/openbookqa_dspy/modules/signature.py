from __future__ import annotations

import dspy


class Answerer(dspy.Signature):
    """Answer the multiple-choice question by returning the single best option (A-D)."""

    question: str = dspy.InputField()
    options: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Only the single letter A, B, C, or D.")
