"""CLI script to query an indexed repository via RepoPilot V1 RAG pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repopilot.config import load_settings
from repopilot.pipeline_v1 import answer_question



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query an indexed code repository with RepoPilot V1 RAG."
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Repository slug identifier (e.g. click).",
    )
    parser.add_argument(
        "question",
        help="Question to ask about the repository.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5).",
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

    # Apply mode override if specified
    if args.server:
        settings.vector_store.mode = "server"
    elif args.local:
        settings.vector_store.mode = "local"

    mode = settings.vector_store.mode
    collection_name = f"{settings.vector_store.collection_prefix}_{args.repo}"

    print(f"[*] Querying Repository: {args.repo}")
    print(f"[*] Qdrant Mode:         {mode}")
    print(f"[*] Collection:          {collection_name}")
    print(f"[*] Question:            {args.question}")
    print("=" * 60)

    try:
        result = answer_question(
            slug=args.repo,
            question=args.question,
            top_k=args.top_k,
            settings=settings,
        )
    except Exception as exc:
        print(f"\n[!] Query failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n=== Answer ===")
    print(result.answer.strip())
    print("\n=== Cited Sources ===")
    if not result.cited_chunks:
        print("(No sources cited)")
    else:
        for idx, chunk in enumerate(result.cited_chunks, start=1):
            citation = chunk.citation
            first_line = chunk.text.strip().splitlines()[0] if chunk.text.strip() else ""
            print(f"[{idx}] {citation} (score: {chunk.score:.4f})")
            print(f"    Preview: {first_line[:90]}")

    print("\n=== Metrics ===")
    print(f"Latency:     {result.latency_ms} ms")
    print(f"Tokens in:   {result.input_tokens}")
    print(f"Tokens out:  {result.output_tokens}")
    print(f"Cache hit:   {result.cache_hit}")
    print("=" * 60)


if __name__ == "__main__":
    main()
