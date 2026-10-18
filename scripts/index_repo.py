"""CLI script to clone, chunk, embed, and index a git repository into Qdrant."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repopilot.config import load_settings
from repopilot.pipeline_v1 import index_repository



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index a git repository into RepoPilot Qdrant vector store."
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Git clone URL (e.g. https://github.com/pallets/click) or local directory.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Repository slug/name identifier (e.g. click).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding and upsert batch size (default: 64).",
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    settings = load_settings()

    # Apply mode override if specified
    if args.server:
        settings.vector_store.mode = "server"
    elif args.local:
        settings.vector_store.mode = "local"

    mode = settings.vector_store.mode
    collection_name = f"{settings.vector_store.collection_prefix}_{args.name}"

    print(f"[*] Starting repository indexing")
    print(f"[*] Target Repository:  {args.repo}")
    print(f"[*] Repository Name:    {args.name}")
    print(f"[*] Qdrant Mode:        {mode}")
    print(f"[*] Target Collection:  {collection_name}")
    if mode == "server":
        print(f"[*] Qdrant Server URL:  {settings.vector_store.url}")
    else:
        print(f"[*] Qdrant Local Path:  {settings.vector_store.path}")
    print("-" * 60)

    try:
        result = index_repository(
            url=args.repo,
            name=args.name,
            settings=settings,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        print(f"\n[!] Indexing failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("-" * 60)
    print("=== Indexing Summary ===")
    print(f"Repository:     {result.repository_url}")
    print(f"Local Path:     {result.dest_path}")
    print(f"Commit SHA:     {result.commit_sha or '<unknown>'}")
    print(f"Qdrant Mode:    {mode}")
    print(f"Collection:     {result.collection_name}")
    print(f"Files Indexed:  {result.files_count}")
    print(f"Chunks Indexed: {result.chunks_count}")
    print(f"Elapsed Time:   {result.elapsed_seconds:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
