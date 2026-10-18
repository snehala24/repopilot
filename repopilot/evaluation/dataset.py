"""Golden evaluation dataset schema and JSONL persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

QuestionCategory = Literal[
    "code_understanding",
    "debugging",
    "change_impact",
    "regression",
    "history",
]


class GoldenQuestion(BaseModel):
    """Schema for a golden evaluation question."""

    id: str = Field(min_length=1, description="Unique identifier for the question.")
    repo: str = Field(min_length=1, description="Repository slug/name identifier.")
    repo_commit: str = Field(
        min_length=1,
        description="Commit SHA the repository was indexed at when question was authored.",
    )
    question: str = Field(min_length=1, description="Question text.")
    category: QuestionCategory = Field(
        description="Evaluation question category.",
    )
    expected_files: list[str] = Field(
        default_factory=list,
        description="Repository-relative POSIX paths of relevant source/doc files.",
    )
    expected_answer_notes: str = Field(
        default="",
        description="Ground truth notes or key points for answering.",
    )
    expected_commits: list[str] | None = Field(
        default=None,
        description="Optional list of commit SHAs relevant to the question.",
    )
    answerable: bool = Field(
        default=True,
        description="Whether the question is answerable from the repository context. Unanswerable questions test refusal.",
    )


def load_golden_dataset(path: Path | str) -> list[GoldenQuestion]:
    """Load and validate a golden dataset from a JSONL file.

    Fails with an error that names the 1-indexed line number on syntax or validation errors.

    Args:
        path: Path to the .jsonl file.

    Returns:
        List of validated :class:`GoldenQuestion` objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If any line contains invalid JSON or schema violations.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Golden dataset file not found: {file_path}")

    questions: list[GoldenQuestion] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue

            try:
                data = json.loads(line_str)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_num} in {file_path}: {exc}"
                ) from exc

            try:
                question = GoldenQuestion.model_validate(data)
                questions.append(question)
            except ValidationError as exc:
                raise ValueError(
                    f"Schema validation error on line {line_num} in {file_path}:\n{exc}"
                ) from exc

    return questions


def save_golden_dataset(
    questions: list[GoldenQuestion],
    path: Path | str,
) -> None:
    """Save a list of GoldenQuestion objects to a JSONL file.

    Args:
        questions: List of GoldenQuestion instances.
        path: Destination file path.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(q.model_dump_json() + "\n")
