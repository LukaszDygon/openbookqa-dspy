from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal
import json
import re

from flask import Flask, render_template, request, redirect

from .config import Settings


# ---- Data models ----


@dataclass(frozen=True)
class QAExample:
    """Single evaluated QA example."""

    index: int
    question: str
    options: list[str]
    expected: str
    predicted: str
    correct: bool


@dataclass(frozen=True)
class EvalFile:
    """Loaded evaluation file with metadata and examples."""

    filename: str
    path: Path
    timestamp: str
    approach: str
    model: str
    sample_size: int
    accuracy: float | None
    correct_count: int
    incorrect_count: int
    examples: list[QAExample]


EvalFilter = Literal["all", "correct", "incorrect"]


# ---- Core logic ----

EVAL_DIR = Path(__file__).resolve().parents[2] / "evaluation_results"
QUESTIONS_PATH = Path(__file__).resolve().parents[2] / "questions.json"
FILENAME_RE = re.compile(r"^(?P<ts>\d{8}_\d{6})_(?P<approach>[a-zA-Z0-9_-]+)-(?:\d+)\.json$")


def _iter_eval_files(directory: Path) -> Iterable[Path]:
    """Yield JSON files from the evaluation directory (sorted newest first)."""
    if not directory.exists():
        return []
    files = sorted(directory.glob("*.json"), key=lambda p: p.name, reverse=True)
    return files


def _parse_filename_meta(path: Path) -> tuple[str, str]:
    """Extract (timestamp, approach) from the evaluation file name.

    If pattern does not match, fall back to ("unknown", stem).
    """
    m = FILENAME_RE.match(path.name)
    if not m:
        return ("unknown", path.stem)
    return (m.group("ts"), m.group("approach"))


def _load_eval(path: Path, model_name: str) -> EvalFile:
    """Load a single evaluation JSON file and enrich with metadata.

    Args:
        path: JSON file path.
        model_name: Model to report (best effort; current configured model).
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    timestamp, approach = _parse_filename_meta(path)
    examples_raw = data.get("examples", [])
    examples: list[QAExample] = [
        QAExample(
            index=int(ex.get("index", -1)),
            question=str(ex.get("question", "")),
            options=[str(o) for o in ex.get("options", [])],
            expected=str(ex.get("expected", "")),
            predicted=str(ex.get("predicted", "")),
            correct=bool(ex.get("correct", False)),
        )
        for ex in examples_raw
    ]

    correct_count = sum(1 for e in examples if e.correct)
    incorrect_count = len(examples) - correct_count

    return EvalFile(
        filename=path.name,
        path=path,
        timestamp=timestamp,
        approach=approach,
        model=model_name,
        sample_size=int(data.get("sample_size", len(examples))),
        accuracy=(float(data["accuracy"]) if "accuracy" in data else None),
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        examples=examples,
    )


def _filter_examples(examples: list[QAExample], flt: EvalFilter) -> list[QAExample]:
    if flt == "correct":
        return [e for e in examples if e.correct]
    if flt == "incorrect":
        return [e for e in examples if not e.correct]
    return examples


# ---- Flask app ----


def create_app() -> Flask:
    """Create and configure Flask app."""
    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))

    @app.get("/")
    def index(): 
        """Render evaluation browser with optional correctness filter."""
        settings = Settings.load()
        model_name = settings.model
        flt: EvalFilter = request.args.get("filter", "all")
        if flt not in ("all", "correct", "incorrect"):
            flt = "all"

        # Determine which file(s) to load
        all_paths = list(_iter_eval_files(EVAL_DIR))
        filenames = [p.name for p in all_paths]
        selected_file = request.args.get("file", "all") or "all"
        paths_to_load: list[Path]
        if selected_file != "all" and selected_file in filenames:
            paths_to_load = [p for p in all_paths if p.name == selected_file]
        else:
            selected_file = "all"
            paths_to_load = all_paths

        evals: list[EvalFile] = [_load_eval(p, model_name=model_name) for p in paths_to_load]

        # Apply example-level filtering per eval
        filtered = [
            EvalFile(
                filename=ev.filename,
                path=ev.path,
                timestamp=ev.timestamp,
                approach=ev.approach,
                model=ev.model,
                sample_size=ev.sample_size,
                accuracy=ev.accuracy,
                correct_count=sum(1 for e in ev.examples if e.correct),
                incorrect_count=sum(1 for e in ev.examples if not e.correct),
                examples=_filter_examples(ev.examples, flt),
            )
            for ev in evals
        ]

        return render_template(
            "results.html",
            filter_value=flt,
            files=filenames,
            selected_file=selected_file,
            evals=filtered,
        )

    @app.get("/questions")
    def questions_get():
        """Render questions.json review UI."""
        items: list[dict[str, object]] = []
        if QUESTIONS_PATH.exists():
            with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
                raw = json.load(f)
                # Ensure list of dicts
                if isinstance(raw, list):
                    for obj in raw:
                        if isinstance(obj, dict):
                            # Initialize review fields if missing
                            obj.setdefault("reviewStatus", "pending")
                            obj.setdefault("rejectionReason", "")
                            items.append(obj)
        return render_template("questions.html", items=items)

    @app.post("/questions")
    def questions_post():
        """Update review fields in questions.json from form submission."""
        if not QUESTIONS_PATH.exists():
            # Nothing to update; redirect back
            return redirect("/questions")

        with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return redirect("/questions")

        # Build index->(status, reason) map from form
        # Inputs are named: status-<index>, reason-<index>
        updated: list[dict[str, object]] = []
        for obj in data:
            if not isinstance(obj, dict):
                continue
            idx = obj.get("index")
            try:
                idx_int = int(idx) if idx is not None else None
            except Exception:
                idx_int = None

            status_key = f"status-{idx_int}" if idx_int is not None else None
            reason_key = f"reason-{idx_int}" if idx_int is not None else None

            status_val = request.form.get(status_key, "pending") if status_key else "pending"
            reason_val = request.form.get(reason_key, "") if reason_key else ""

            obj["reviewStatus"] = status_val
            obj["rejectionReason"] = reason_val if status_val == "rejected" else ""
            updated.append(obj)

        # Write back pretty-printed JSON
        QUESTIONS_PATH.write_text(json.dumps(updated, ensure_ascii=False, indent=2))
        return redirect("/questions")

    return app


def main() -> None:
    """Run development server."""
    app = create_app()
    # Host 127.0.0.1 to avoid external exposure by default
    app.run(host="127.0.0.1", port=5000, debug=True)
