from __future__ import annotations

"""Shared LM configuration utilities for DSPy."""

import os

import dspy

from .config import Settings


def configure_dspy_lm(settings: Settings) -> None:
    """Configure DSPy to use its built-in LM with OpenAI provider.

    Sets the OPENAI_API_KEY if provided in settings and initializes the LM.
    """
    if settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key

    lm = dspy.LM(
        model=settings.model,
        cache=False,
    )
    dspy.settings.configure(lm=lm)
