from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from environment.

    Attributes:
        openai_api_key: API key for OpenAI.
        hf_home: Optional Hugging Face cache directory.
        model: OpenAI model name to use (default: openai/gpt-4o-mini).
    """

    openai_api_key: str
    hf_home: str | None = None
    model: str = "openai/gpt-4o-mini"

    @staticmethod
    def load() -> "Settings":
        """Load settings from environment variables.

        Returns:
            Loaded `Settings` instance.
        """
        return Settings(
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            hf_home=os.environ.get("HF_HOME"),
            model=os.environ.get("OPENBOOKQA_DSPY_MODEL", "openai/gpt-4o-mini"),
        )
