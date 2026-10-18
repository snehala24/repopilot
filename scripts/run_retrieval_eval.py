"""CLI script to run retrieval evaluation against golden datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repopilot.config import load_settings
from repopilot.evaluation.runner import run_retrieval_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run retrieval evaluation benchmarks on a golden dataset."
    )
    parser.add_argument(
        "--retriever",
        choices=["dense"],
        default="dense",
        help="Retriever implementation to evaluate (default: dense).",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to golden dataset JSONL file (e.g. eval_samples/golden_sample.jsonl).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-K cutoff for retrieval metrics (default: 5).",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--server",
        action="store_true",
        help="Override vector store mode to server (connects to Qdrant service URL).",
    )
    mode_group.add_argument(
        "--local",
        action="store_true",
        help="Override vector store mode to local (embedded storage in data/qdrant).",
    )

    args = parser.parse_args()

    settings = load_settings()
    if args.server:
        settings.vector_store.mode = "server"
    elif args.local:
        settings.vector_store.mode = "local"

    print("=" * 65)
    print(f"[*] Starting Retrieval Evaluation")
    print(f"[*] Dataset:    {args.dataset}")
    print(f"[*] Retriever:  {args.retriever}")
    print(f"[*] Top-K:      {args.k}")
    print(f"[*] Store Mode: {settings.vector_store.mode}")
    print("=" * 65)

    try:
        summary = run_retrieval_evaluation(
            dataset_path=args.dataset,
            retriever_name=args.retriever,
            settings=settings,
            k=args.k,
        )
    except Exception as exc:
        print(f"\n[!] Evaluation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n=== Overall Retrieval Metrics ===")
    print(f"Evaluated Questions:  {summary.evaluated_questions} (Skipped unanswerable: {summary.skipped_unanswerable})")
    print(f"Precision@{args.k}:        {summary.mean_precision_at_k:.4f}")
    print(f"Recall@{args.k}:           {summary.mean_recall_at_k:.4f}")
    print(f"MRR:                  {summary.mean_mrr:.4f}")
    print(f"NDCG@{args.k}:            {summary.mean_ndcg_at_k:.4f}")

    print("\n=== Source-Type Distribution (Top-K Chunks) ===")
    print(f"Code Share:           {summary.mean_code_share * 100:.1f}%")
    print(f"Docs Share:           {summary.mean_docs_share * 100:.1f}%")
    print(f"Tests Share:          {summary.mean_tests_share * 100:.1f}%")

    print("\n=== Performance by Category ===")
    print(f"{'Category':<20} | {'Count':<5} | {'P@K':<7} | {'R@K':<7} | {'MRR':<7} | {'NDCG@K':<7}")
    print("-" * 65)
    for cat, m in sorted(summary.category_metrics.items()):
        print(
            f"{cat:<20} | {int(m['count']):<5} | {m['precision_at_k']:<7.4f} | {m['recall_at_k']:<7.4f} | {m['mrr']:<7.4f} | {m['ndcg_at_k']:<7.4f}"
        )

    print("\n" + "=" * 65)
    print(f"[*] Results saved to: {summary.output_dir.relative_to(PROJECT_ROOT)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
