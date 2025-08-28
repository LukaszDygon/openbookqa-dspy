from __future__ import annotations

from typing import Any
import dspy
from openai import OpenAI
from .config import Settings
from .data import QAExample


class OpenAICompletion(dspy.LM):
    """DSPy LM wrapper around OpenAI chat completions for simple completion."""

    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model=model, cache=False)
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        """Return a text completion for either a prompt or chat messages.

        Supports both:
        - prompt: str
        - messages: list[{"role": str, "content": str}] (preferred by DSPy adapters)
        """
        # Extract possible inputs
        prompt: str | None = kwargs.pop("prompt", None)
        messages: list[dict[str, str]] | None = kwargs.pop("messages", None)

        if messages is None:
            # Some callers may pass the prompt as a positional arg
            if prompt is None and args:
                maybe_prompt = args[0]
                if isinstance(maybe_prompt, str):
                    prompt = maybe_prompt
            if prompt is None:
                raise TypeError(
                    "OpenAICompletion.__call__ requires 'messages' or 'prompt' argument"
                )
            messages = [{"role": "user", "content": prompt}]

        temperature = float(kwargs.pop("temperature", 0.0))
        max_tokens = int(kwargs.pop("max_tokens", 16))

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


def build_pipeline(settings: Settings) -> dspy.Module:
    """Create a minimal DSPy pipeline that returns a single letter (A-D).

    This implementation avoids DSPy's JSONAdapter by constructing a strict
    prompt and parsing the letter locally.
    """
    lm = OpenAICompletion(model=settings.model, api_key=settings.openai_api_key)
    dspy.settings.configure(lm=lm)

    class MCQAnswerer(dspy.Module):
        def forward(self, question: str, options: str):  # type: ignore[override]
            import re

            prompt = (
                "You are a careful multiple-choice solver.\n"
                "Choose the best single option.\n"
                "Question: "
                f"{question}\n"
                "Options:\n"
                f"{options}\n\n"
                "Respond with ONLY one character: A, B, C, or D. No words."
            )
            text = dspy.settings.lm(prompt=prompt, max_tokens=4, temperature=1.0)  # type: ignore[call-arg]
            m = re.search(r"[ABCD]", str(text).upper())
            ans = m.group(0) if m else ""
            return dspy.Prediction(answer=ans)  # type: ignore[no-any-return]

    return MCQAnswerer()


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
