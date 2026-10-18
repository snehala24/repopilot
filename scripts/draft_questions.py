"""CLI script to draft candidate evaluation questions from an indexed repository.

Samples chunks stratified across code, docs, and tests in roughly equal proportions,
invokes get_llm('strong') to draft candidate questions and hint files, and saves
unreviewed questions to data/eval/candidates_<slug>.jsonl with reviewed=false.

CRITICAL:
- Candidate expected_files are HINTS ONLY and must be verified by a human.
- Never automatically merge drafted candidates into the golden dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repopilot.config import Settings, load_settings
from repopilot.evaluation.metrics import classify_source_type
from repopilot.evaluation.runner import get_collection_commit_sha
from repopilot.llm import get_llm
from repopilot.retrieval.store import QdrantVectorStore

logger = logging.getLogger(__name__)


class CandidateDraft(BaseModel):
    """Structured LLM response for a drafted candidate question."""

    question: str = Field(
        description="A natural developer question that can be answered from this code or documentation."
    )
    category: Literal[
        "code_understanding",
        "debugging",
        "change_impact",
        "regression",
        "history",
    ] = Field(description="Question category.")
    hint_files: list[str] = Field(
        default_factory=list,
        description="Proposed relevant repository files (HINTS ONLY, pending human review).",
    )
    answer_notes: str = Field(
        description="Concise notes on the ground truth answer based on the sampled chunk."
    )


DRAFT_SYSTEM_PROMPT = """\
You are an expert AI dataset curator designing a high-quality evaluation benchmark for a code assistant.
Given a code or documentation snippet from a repository, draft ONE challenging, realistic developer question.

RULES:
1. The question must be realistic (something a software engineer working on this repository would ask).
2. Choose the most appropriate category: code_understanding, debugging, change_impact, regression, or history.
3. In hint_files, list the seed file and any other likely relevant files mentioned in the snippet. (Note: These are preliminary hints only).
4. In answer_notes, summarize the core facts from the snippet that answer the question.\
"""


def draft_questions_from_repo(
    repo_slug: str,
    n: int = 20,
    *,
    server: bool = False,
    local: bool = False,
    output_path: Path | None = None,
    store: QdrantVectorStore | None = None,
    llm: Any = None,
    settings: Settings | None = None,
) -> Path:
    """Sample chunks stratified across code, docs, and tests, and draft candidate questions."""
    if settings is None:
        settings = load_settings()
    if server:
        settings.vector_store.mode = "server"
    elif local:
        settings.vector_store.mode = "local"

    own_store = False
    if store is None:
        store = QdrantVectorStore.from_settings(settings)
        own_store = True

    collection_name = f"{settings.vector_store.collection_prefix}_{repo_slug}"

    try:
        commit_sha = get_collection_commit_sha(store, collection_name)
        logger.info("Sampling chunks from collection %s (commit %s)...", collection_name, commit_sha)

        # 1. Fetch points pool from Qdrant
        records, _ = store.client.scroll(
            collection_name=collection_name,
            limit=250,
            with_payload=True,
        )

        if not records:
            raise ValueError(f"Collection {collection_name} is empty.")

        # 2. Stratify points into code, docs, tests (Adjustment 4)
        stratified: dict[str, list[dict]] = {"code": [], "docs": [], "tests": []}
        for r in records:
            payload = r.payload or {}
            file_path = payload.get("file_path", "")
            st = classify_source_type(file_path)
            stratified[st].append(payload)

        logger.info(
            "Available points pool: %d code, %d docs, %d tests",
            len(stratified["code"]),
            len(stratified["docs"]),
            len(stratified["tests"]),
        )

        # Stratified sampling in roughly equal proportions
        target_per_bucket = (n + 2) // 3
        selected_payloads: list[tuple[str, dict]] = []

        for bucket_name in ["code", "docs", "tests"]:
            pool = stratified[bucket_name]
            k_samples = min(len(pool), target_per_bucket)
            sampled = random.sample(pool, k_samples) if pool else []
            for p in sampled:
                selected_payloads.append((bucket_name, p))

        # Shuffle combined seeds and truncate to n
        random.shuffle(selected_payloads)
        selected_payloads = selected_payloads[:n]

        logger.info("Selected %d seed chunks for drafting candidate questions.", len(selected_payloads))

        # 3. LLM Gateway (strong tier) respecting budget guard and cache
        if llm is None:
            llm = get_llm("strong", settings)

        out_file = output_path or (PROJECT_ROOT / "data" / "eval" / f"candidates_{repo_slug}.jsonl")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        candidates: list[dict] = []

        for idx, (source_type, payload) in enumerate(selected_payloads, start=1):
            file_path = payload.get("file_path", "unknown")
            line_start = payload.get("line_start", 0)
            line_end = payload.get("line_end", 0)
            content = payload.get("content", "")

            user_prompt = (
                f"Repository: {repo_slug}\n"
                f"File: {file_path} (lines {line_start}-{line_end})\n"
                f"Source type: {source_type}\n\n"
                f"Content snippet:\n```\n{content}\n```\n\n"
                f"Draft a candidate developer evaluation question based on this snippet."
            )

            try:
                draft: CandidateDraft = llm.generate_structured(
                    prompt=user_prompt,
                    response_model=CandidateDraft,
                    system=DRAFT_SYSTEM_PROMPT,
                )
            except Exception as exc:
                logger.warning("Failed to draft question %d/%d: %s", idx, n, exc)
                continue

            # Ensure seed file is in hint_files
            hints = list(draft.hint_files)
            if file_path not in hints:
                hints.insert(0, file_path)

            record = {
                "id": f"{repo_slug}-candidate-{idx:03d}",
                "repo": repo_slug,
                "repo_commit": commit_sha,
                "question": draft.question,
                "category": draft.category,
                "expected_files": hints,  # HINTS ONLY
                "expected_answer_notes": (
                    f"[UNREVIEWED CANDIDATE - HINTS ONLY] {draft.answer_notes}"
                ),
                "expected_commits": None,
                "answerable": True,
                "reviewed": False,
                "source_seed_file": file_path,
                "source_seed_type": source_type,
            }
            candidates.append(record)
            logger.info("Drafted question %d/%d: [%s] %s", idx, len(selected_payloads), draft.category, draft.question[:60])

        # 4. Save to JSONL
        with out_file.open("w", encoding="utf-8") as f:
            for c in candidates:
                f.write(json.dumps(c) + "\n")

        logger.info("Successfully wrote %d unreviewed candidate questions to %s", len(candidates), out_file)
        return out_file
    finally:
        if own_store:
            store.close()



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draft candidate evaluation questions from an indexed repository."
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Repository slug identifier (e.g. click).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of candidate questions to draft (default: 20).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Custom output JSONL file path (default: data/eval/candidates_<slug>.jsonl).",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--server",
        action="store_true",
        help="Override vector store mode to server.",
    )
    mode_group.add_argument(
        "--local",
        action="store_true",
        help="Override vector store mode to local.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 65)
    print(f"[*] Drafting Candidate Evaluation Questions")
    print(f"[*] Target Repo: {args.repo}")
    print(f"[*] Count (n):   {args.n}")
    print("=" * 65)

    try:
        out_path = draft_questions_from_repo(
            repo_slug=args.repo,
            n=args.n,
            server=args.server,
            local=args.local,
            output_path=args.output,
        )
    except Exception as exc:
        print(f"\n[!] Question drafting failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("-" * 65)
    print(f"[+] Candidate questions written to: {out_path}")
    print(f"[!] NOTICE: All candidate questions have reviewed=false.")
    print(f"[!] NOTICE: expected_files are AI-generated HINTS ONLY.")
    print(f"[!] A human curator must review, verify files, and approve before adding to golden datasets.")
    print("=" * 65)


if __name__ == "__main__":
    main()
