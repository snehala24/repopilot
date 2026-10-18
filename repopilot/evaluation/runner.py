"""Evaluation runner for repository retrieval pipelines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import git

from repopilot.config import PROJECT_ROOT, Settings, load_settings
from repopilot.evaluation.dataset import GoldenQuestion, load_golden_dataset
from repopilot.evaluation.metrics import (
    RetrievalMetrics,
    compute_retrieval_metrics,
)
from repopilot.retrieval.dense import DenseRetriever
from repopilot.retrieval.store import QdrantVectorStore

if TYPE_CHECKING:
    from repopilot.retrieval.base import Retriever


@dataclass
class QuestionEvalResult:
    """Evaluation result for a single golden question."""

    id: str
    question: str
    category: str
    expected_files: list[str]
    retrieved_unique_files: list[str]
    retrieved_chunks: list[dict[str, Any]]
    metrics: RetrievalMetrics


@dataclass
class EvaluationSummary:
    """Aggregated evaluation metrics across a dataset."""

    timestamp: str
    git_sha: str
    retriever_name: str
    k: int
    collection_name: str
    repo_commit: str
    total_questions: int
    evaluated_questions: int
    skipped_unanswerable: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_mrr: float
    mean_ndcg_at_k: float
    mean_code_share: float
    mean_docs_share: float
    mean_tests_share: float
    category_metrics: dict[str, dict[str, float]]
    question_results: list[QuestionEvalResult]
    output_dir: Path


def get_collection_commit_sha(
    store: QdrantVectorStore, collection_name: str
) -> str:
    """Retrieve the commit_sha metadata stored in an indexed Qdrant collection."""
    if not store.client.collection_exists(collection_name):
        raise ValueError(
            f"Collection {collection_name!r} does not exist in Qdrant store."
        )

    records, _ = store.client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=True,
    )
    if not records:
        raise ValueError(
            f"Collection {collection_name!r} is empty. Index the repository before evaluating."
        )

    payload = records[0].payload or {}
    return str(payload.get("commit_sha", ""))


def get_current_git_sha() -> str:
    """Return the current repository HEAD commit SHA."""
    try:
        repo = git.Repo(PROJECT_ROOT)
        return repo.head.commit.hexsha
    except Exception:
        return "unknown"


def run_retrieval_evaluation(
    dataset_path: Path | str,
    *,
    retriever_name: str = "dense",
    retriever: Retriever | None = None,
    store: QdrantVectorStore | None = None,
    settings: Settings | None = None,
    k: int = 5,
    results_base_dir: Path | None = None,
) -> EvaluationSummary:
    """Run retrieval evaluation over a golden dataset.

    Args:
        dataset_path: Path to the golden dataset JSONL file.
        retriever_name: Identifier for the retriever ('dense').
        retriever: Optional injected Retriever instance (for tests).
        store: Optional injected QdrantVectorStore instance.
        settings: Application settings.
        k: Top-K cutoff.
        results_base_dir: Directory where timestamped results will be saved.

    Returns:
        :class:`EvaluationSummary` with aggregated metrics.

    Raises:
        ValueError: If commit mismatch is detected or dataset is empty.
    """
    settings = settings or load_settings()
    questions = load_golden_dataset(dataset_path)
    if not questions:
        raise ValueError(f"Golden dataset at {dataset_path} contains no questions.")

    # Target repository slug from dataset
    repo_slug = questions[0].repo
    collection_name = f"{settings.vector_store.collection_prefix}_{repo_slug}"

    if store is None:
        store = QdrantVectorStore.from_settings(settings)

    # Validate commit matching against indexed collection payload
    collection_commit = get_collection_commit_sha(store, collection_name)
    for q in questions:
        if q.repo_commit != collection_commit:
            raise ValueError(
                f"Commit mismatch for question '{q.id}': question was authored against commit "
                f"'{q.repo_commit}', but collection '{collection_name}' is indexed at commit "
                f"'{collection_commit}'. Evaluation aborted. Re-index the repository or update question commit."
            )

    # Initialize retriever if not injected
    if retriever is None:
        if retriever_name == "dense":
            retriever = DenseRetriever.from_settings(
                settings, collection_name=collection_name, vector_store=store
            )
        else:
            raise ValueError(f"Unsupported retriever: {retriever_name!r}")

    # Separate answerable vs unanswerable questions (Adjustment 2)
    eval_questions: list[GoldenQuestion] = [q for q in questions if q.answerable]
    skipped_unanswerable = len(questions) - len(eval_questions)

    question_results: list[QuestionEvalResult] = []
    category_buckets: dict[str, list[RetrievalMetrics]] = {}

    for q in eval_questions:
        chunks = retriever.retrieve(q.question, top_k=k)
        metrics = compute_retrieval_metrics(chunks, q.expected_files, k=k)

        chunk_payloads = [
            {
                "chunk_id": c.chunk_id,
                "file_path": c.file_path,
                "line_start": c.line_start,
                "line_end": c.line_end,
                "score": c.score,
            }
            for c in chunks[:k]
        ]

        res = QuestionEvalResult(
            id=q.id,
            question=q.question,
            category=q.category,
            expected_files=q.expected_files,
            retrieved_unique_files=metrics.retrieved_unique_files[:k],
            retrieved_chunks=chunk_payloads,
            metrics=metrics,
        )
        question_results.append(res)

        if q.category not in category_buckets:
            category_buckets[q.category] = []
        category_buckets[q.category].append(metrics)

    n_eval = len(question_results)
    mean_p = sum(r.metrics.precision_at_k for r in question_results) / n_eval if n_eval else 0.0
    mean_r = sum(r.metrics.recall_at_k for r in question_results) / n_eval if n_eval else 0.0
    mean_mrr = sum(r.metrics.mrr for r in question_results) / n_eval if n_eval else 0.0
    mean_ndcg = sum(r.metrics.ndcg_at_k for r in question_results) / n_eval if n_eval else 0.0

    mean_code = (
        sum(r.metrics.source_distribution.code_share for r in question_results) / n_eval
        if n_eval
        else 0.0
    )
    mean_docs = (
        sum(r.metrics.source_distribution.docs_share for r in question_results) / n_eval
        if n_eval
        else 0.0
    )
    mean_tests = (
        sum(r.metrics.source_distribution.tests_share for r in question_results) / n_eval
        if n_eval
        else 0.0
    )

    category_metrics: dict[str, dict[str, float]] = {}
    for cat, m_list in category_buckets.items():
        c_len = len(m_list)
        category_metrics[cat] = {
            "count": float(c_len),
            "precision_at_k": sum(m.precision_at_k for m in m_list) / c_len,
            "recall_at_k": sum(m.recall_at_k for m in m_list) / c_len,
            "mrr": sum(m.mrr for m in m_list) / c_len,
            "ndcg_at_k": sum(m.ndcg_at_k for m in m_list) / c_len,
        }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    git_sha = get_current_git_sha()
    base_dir = results_base_dir or (PROJECT_ROOT / "results")
    output_dir = base_dir / f"{timestamp}_{retriever_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = EvaluationSummary(
        timestamp=timestamp,
        git_sha=git_sha,
        retriever_name=retriever_name,
        k=k,
        collection_name=collection_name,
        repo_commit=collection_commit,
        total_questions=len(questions),
        evaluated_questions=n_eval,
        skipped_unanswerable=skipped_unanswerable,
        mean_precision_at_k=mean_p,
        mean_recall_at_k=mean_r,
        mean_mrr=mean_mrr,
        mean_ndcg_at_k=mean_ndcg,
        mean_code_share=mean_code,
        mean_docs_share=mean_docs,
        mean_tests_share=mean_tests,
        category_metrics=category_metrics,
        question_results=question_results,
        output_dir=output_dir,
    )

    _write_results_json(summary, settings, output_dir / "results.json")
    _write_summary_md(summary, output_dir / "summary.md")

    return summary


def _write_results_json(
    summary: EvaluationSummary, settings: Settings, out_path: Path
) -> None:
    """Serialize full evaluation details and config snapshot to JSON."""
    data = {
        "metadata": {
            "timestamp": summary.timestamp,
            "git_sha": summary.git_sha,
            "retriever": summary.retriever_name,
            "k": summary.k,
            "collection_name": summary.collection_name,
            "repo_commit": summary.repo_commit,
            "total_questions": summary.total_questions,
            "evaluated_questions": summary.evaluated_questions,
            "skipped_unanswerable": summary.skipped_unanswerable,
        },
        "mean_metrics": {
            "precision_at_k": round(summary.mean_precision_at_k, 4),
            "recall_at_k": round(summary.mean_recall_at_k, 4),
            "mrr": round(summary.mean_mrr, 4),
            "ndcg_at_k": round(summary.mean_ndcg_at_k, 4),
        },
        "source_type_distribution": {
            "code_share": round(summary.mean_code_share, 4),
            "docs_share": round(summary.mean_docs_share, 4),
            "tests_share": round(summary.mean_tests_share, 4),
        },
        "category_metrics": {
            cat: {k: round(v, 4) for k, v in m.items()}
            for cat, m in summary.category_metrics.items()
        },
        "config_snapshot": {
            "ingestion": {
                "chunk_size": settings.ingestion.chunk_size,
                "chunk_overlap": settings.ingestion.chunk_overlap,
                "max_file_size_bytes": settings.ingestion.max_file_size_bytes,
            },
            "embedding": {
                "model": settings.embedding.model,
                "dimensions": settings.embedding.dimensions,
            },
            "vector_store": {
                "mode": settings.vector_store.mode,
                "url": settings.vector_store.url,
            },
        },
        "questions": [
            {
                "id": r.id,
                "question": r.question,
                "category": r.category,
                "expected_files": r.expected_files,
                "retrieved_unique_files": r.retrieved_unique_files,
                "metrics": {
                    "precision_at_k": round(r.metrics.precision_at_k, 4),
                    "recall_at_k": round(r.metrics.recall_at_k, 4),
                    "mrr": round(r.metrics.mrr, 4),
                    "ndcg_at_k": round(r.metrics.ndcg_at_k, 4),
                    "source_distribution": asdict(r.metrics.source_distribution),
                },
                "retrieved_chunks": r.retrieved_chunks,
            }
            for r in summary.question_results
        ],
    }

    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_summary_md(summary: EvaluationSummary, out_path: Path) -> None:
    """Write human-readable markdown evaluation report."""
    # Find 5 worst questions based on NDCG, MRR, Recall
    worst = sorted(
        summary.question_results,
        key=lambda r: (r.metrics.ndcg_at_k, r.metrics.mrr, r.metrics.recall_at_k),
    )[:5]

    lines = [
        f"# Retrieval Evaluation Summary: `{summary.retriever_name}` (K={summary.k})",
        "",
        f"- **Timestamp**: `{summary.timestamp}`",
        f"- **Repo Commit**: `{summary.repo_commit}`",
        f"- **RepoPilot Git SHA**: `{summary.git_sha}`",
        f"- **Collection**: `{summary.collection_name}`",
        f"- **Questions Evaluated**: {summary.evaluated_questions} (Total: {summary.total_questions}, Skipped unanswerable: {summary.skipped_unanswerable})",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Score |",
        "| :--- | :--- |",
        f"| **Precision@{summary.k}** | {summary.mean_precision_at_k:.4f} |",
        f"| **Recall@{summary.k}** | {summary.mean_recall_at_k:.4f} |",
        f"| **MRR** | {summary.mean_mrr:.4f} |",
        f"| **NDCG@{summary.k}** | {summary.mean_ndcg_at_k:.4f} |",
        "",
        "## Source-Type Distribution (Top-K Chunks)",
        "",
        "| Source Type | Share |",
        "| :--- | :--- |",
        f"| Code (`src/` & non-test source) | {summary.mean_code_share * 100:.1f}% |",
        f"| Docs (`docs/` & markdown) | {summary.mean_docs_share * 100:.1f}% |",
        f"| Tests (`tests/`) | {summary.mean_tests_share * 100:.1f}% |",
        "",
        "## Performance by Category",
        "",
        "| Category | Count | Precision@K | Recall@K | MRR | NDCG@K |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for cat, m in sorted(summary.category_metrics.items()):
        lines.append(
            f"| `{cat}` | {int(m['count'])} | {m['precision_at_k']:.4f} | {m['recall_at_k']:.4f} | {m['mrr']:.4f} | {m['ndcg_at_k']:.4f} |"
        )

    lines.extend([
        "",
        "## 5 Worst-Performing Questions",
        "",
    ])

    if not worst:
        lines.append("_No queries evaluated._")
    else:
        for idx, w in enumerate(worst, start=1):
            expected_str = ", ".join(f"`{f}`" for f in w.expected_files) or "<none>"
            retrieved_str = ", ".join(f"`{f}`" for f in w.retrieved_unique_files) or "<none>"
            lines.extend([
                f"### {idx}. [{w.id}] {w.question}",
                f"- **Category**: `{w.category}`",
                f"- **Scores**: NDCG@{summary.k} = `{w.metrics.ndcg_at_k:.4f}` | MRR = `{w.metrics.mrr:.4f}` | Recall@{summary.k} = `{w.metrics.recall_at_k:.4f}` | Precision@{summary.k} = `{w.metrics.precision_at_k:.4f}`",
                f"- **Expected Files**: {expected_str}",
                f"- **Retrieved Files**: {retrieved_str}",
                "",
            ])

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
