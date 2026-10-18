"""Information retrieval metrics for repository search evaluation.

Evaluates file-level relevance over ranked chunks and ranked unique files.
Zero dependencies on external ML/eval frameworks (pure Python and math).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from repopilot.retrieval.base import RetrievedChunk

SourceType = Literal["code", "docs", "tests"]


@dataclass(frozen=True)
class SourceTypeBreakdown:
    """Distribution of top-K chunks across source categories."""

    code_share: float
    docs_share: float
    tests_share: float


@dataclass(frozen=True)
class RetrievalMetrics:
    """Metrics computed for a single query's retrieval results."""

    k: int
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    source_distribution: SourceTypeBreakdown
    retrieved_unique_files: list[str]


def classify_source_type(file_path: str) -> SourceType:
    """Classify a relative repository file path into code, docs, or tests.

    Rules:
    - If path starts with or contains 'tests/', starts with 'test_', or ends with '_test.py' -> 'tests'.
    - Else if path starts with or contains 'docs/', or ends in '.md'/'.rst' -> 'docs'.
    - Otherwise -> 'code' (source files, modules, entry points, configuration).
    """
    p = Path(file_path).as_posix().lower()
    parts = p.split("/")
    filename = parts[-1] if parts else ""

    if (
        "tests" in parts
        or filename.startswith("test_")
        or filename.endswith("_test.py")
    ):
        return "tests"

    if (
        "docs" in parts
        or filename.endswith(".md")
        or filename.endswith(".rst")
    ):
        return "docs"

    return "code"


def compute_source_breakdown(
    chunks: list[RetrievedChunk], k: int
) -> SourceTypeBreakdown:
    """Compute share of code, docs, and tests in top-K chunks."""
    top_chunks = chunks[:k]
    if not top_chunks:
        return SourceTypeBreakdown(code_share=0.0, docs_share=0.0, tests_share=0.0)

    total = len(top_chunks)
    counts = {"code": 0, "docs": 0, "tests": 0}
    for c in top_chunks:
        st = classify_source_type(c.file_path)
        counts[st] += 1

    return SourceTypeBreakdown(
        code_share=counts["code"] / total,
        docs_share=counts["docs"] / total,
        tests_share=counts["tests"] / total,
    )


def compute_retrieval_metrics(
    chunks: list[RetrievedChunk],
    expected_files: list[str],
    *,
    k: int = 5,
) -> RetrievalMetrics:
    """Compute Precision@K, Recall@K, MRR, and NDCG@K against expected files.

    Relevance is file-level: a chunk is relevant if its file path is in expected_files.
    - Precision@K is computed on the ranked CHUNK list.
    - Recall@K, MRR, and NDCG@K are computed on the ranked list of UNIQUE files
      (preserving first occurrence order).

    Args:
        chunks: List of RetrievedChunk objects in descending order of retriever score.
        expected_files: List of expected ground-truth repository-relative file paths.
        k: Cutoff rank (must be > 0).

    Returns:
        :class:`RetrievalMetrics` with computed metrics.

    Raises:
        ValueError: If expected_files is empty or k is non-positive.
    """
    if not expected_files:
        raise ValueError(
            "expected_files cannot be empty for retrieval evaluation. "
            "Use answerable=False for questions with no expected files."
        )
    if k <= 0:
        raise ValueError(f"k must be strictly positive, got {k}")

    norm_expected = {Path(f).as_posix().lower() for f in expected_files}
    source_dist = compute_source_breakdown(chunks, k)

    # 1. Precision@K: on top-K CHUNKS
    top_chunks = chunks[:k]
    relevant_chunks = sum(
        1 for c in top_chunks if Path(c.file_path).as_posix().lower() in norm_expected
    )
    precision_at_k = relevant_chunks / k

    # If no results returned at all
    if not chunks:
        return RetrievalMetrics(
            k=k,
            precision_at_k=0.0,
            recall_at_k=0.0,
            mrr=0.0,
            ndcg_at_k=0.0,
            source_distribution=source_dist,
            retrieved_unique_files=[],
        )

    # 2. Extract ranked list of unique files (first occurrence rank preserved)
    unique_files: list[str] = []
    seen_files: set[str] = set()
    for c in chunks:
        norm_p = Path(c.file_path).as_posix().lower()
        if norm_p and norm_p not in seen_files:
            seen_files.add(norm_p)
            unique_files.append(norm_p)

    top_unique_files = unique_files[:k]

    # 3. Recall@K on top-K unique files
    relevant_unique_files = sum(
        1 for f in top_unique_files if f in norm_expected
    )
    recall_at_k = relevant_unique_files / len(norm_expected)

    # 4. MRR on top-K unique files (reciprocal rank of first relevant file)
    mrr = 0.0
    for rank, f in enumerate(top_unique_files, start=1):
        if f in norm_expected:
            mrr = 1.0 / rank
            break

    # 5. NDCG@K on top-K unique files (binary relevance)
    dcg = 0.0
    for rank, f in enumerate(top_unique_files, start=1):
        rel = 1.0 if f in norm_expected else 0.0
        if rel > 0:
            dcg += rel / math.log2(rank + 1)

    ideal_count = min(k, len(norm_expected))
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_count + 1))
    ndcg_at_k = (dcg / idcg) if idcg > 0 else 0.0

    return RetrievalMetrics(
        k=k,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        mrr=mrr,
        ndcg_at_k=ndcg_at_k,
        source_distribution=source_dist,
        retrieved_unique_files=unique_files,
    )
