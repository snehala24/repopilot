"""Comprehensive unit tests for evaluation metrics, dataset schema, runner, and drafter."""

import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import git
import pytest

from repopilot.config import (
    AppSettings,
    BudgetSettings,
    EmbeddingSettings,
    Environment,
    IngestionSettings,
    LLMSettings,
    LLMTierSettings,
    PathsSettings,
    Settings,
    VectorStoreSettings,
)
from repopilot.evaluation.dataset import (
    GoldenQuestion,
    load_golden_dataset,
    save_golden_dataset,
)
from repopilot.evaluation.metrics import (
    classify_source_type,
    compute_retrieval_metrics,
    compute_source_breakdown,
)
from repopilot.evaluation.runner import run_retrieval_evaluation
from repopilot.pipeline_v1 import index_repository
from repopilot.retrieval.base import RetrievedChunk
from repopilot.retrieval.embedder import FakeEmbedder
from repopilot.retrieval.store import QdrantVectorStore


def make_chunk(file_path: str, score: float = 0.9) -> RetrievedChunk:
    """Helper to construct a RetrievedChunk for metric testing."""
    return RetrievedChunk(
        chunk_id=f"id-{file_path}",
        text=f"content from {file_path}",
        score=score,
        metadata={"file_path": file_path},
    )


# =========================================================================
# 1. Retrieval Metrics Hand-Computed Test Vectors
# =========================================================================


def test_metrics_normal_case():
    """Verify Precision@K on chunks, and Recall@K, MRR, NDCG@K on unique files."""
    expected = ["src/a.py", "src/b.py"]
    chunks = [
        make_chunk("src/a.py", 0.95),      # rank 1 (rel)
        make_chunk("src/c.py", 0.90),      # rank 2 (non-rel)
        make_chunk("src/b.py", 0.85),      # rank 3 (rel)
        make_chunk("src/d.py", 0.80),      # rank 4
        make_chunk("src/e.py", 0.75),      # rank 5
    ]

    m = compute_retrieval_metrics(chunks, expected, k=3)

    # Precision@3 on chunks: [a, c, b] -> 2 relevant chunks / 3 = 2/3
    assert math.isclose(m.precision_at_k, 2 / 3, rel_tol=1e-4)

    # Unique files in top-3: [src/a.py, src/c.py, src/b.py]
    # Recall@3: 2 found / 2 expected = 1.0
    assert math.isclose(m.recall_at_k, 1.0, rel_tol=1e-4)

    # MRR: first relevant file is src/a.py at rank 1 -> 1.0
    assert math.isclose(m.mrr, 1.0, rel_tol=1e-4)

    # NDCG@3: DCG = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.63092975 = 1.63092975
    # NDCG = 1.5 / 1.63092975 = 0.91972...
    expected_ndcg = 1.5 / (1.0 + 1.0 / math.log2(3))
    assert math.isclose(m.ndcg_at_k, expected_ndcg, rel_tol=1e-4)


def test_metrics_duplicate_chunks_from_same_file():
    """Duplicate chunks from the same file must be deduplicated for file metrics."""
    expected = ["src/a.py"]
    chunks = [
        make_chunk("src/a.py", 0.95),
        make_chunk("src/a.py", 0.90),
        make_chunk("src/other.py", 0.80),
    ]

    m = compute_retrieval_metrics(chunks, expected, k=2)

    # Precision@2 on chunks: both chunks are from src/a.py -> 2/2 = 1.0
    assert math.isclose(m.precision_at_k, 1.0, rel_tol=1e-4)

    # Unique files: [src/a.py, src/other.py]
    # Recall@2: 1 / 1 = 1.0
    assert math.isclose(m.recall_at_k, 1.0, rel_tol=1e-4)
    # MRR: rank 1 -> 1.0
    assert math.isclose(m.mrr, 1.0, rel_tol=1e-4)
    # NDCG@2: DCG = 1.0, IDCG = 1.0 -> 1.0
    assert math.isclose(m.ndcg_at_k, 1.0, rel_tol=1e-4)


def test_metrics_no_relevant_results():
    """When no retrieved files match expected, all metrics should be 0.0."""
    expected = ["src/target.py"]
    chunks = [
        make_chunk("src/foo.py", 0.9),
        make_chunk("src/bar.py", 0.8),
    ]

    m = compute_retrieval_metrics(chunks, expected, k=5)
    assert m.precision_at_k == 0.0
    assert m.recall_at_k == 0.0
    assert m.mrr == 0.0
    assert m.ndcg_at_k == 0.0


def test_metrics_k_larger_than_results():
    """When K is larger than the number of retrieved chunks, metrics handle gracefully."""
    expected = ["src/a.py", "src/b.py"]
    chunks = [make_chunk("src/a.py", 0.9)]  # only 1 result

    m = compute_retrieval_metrics(chunks, expected, k=5)
    # Precision@5: 1 relevant / 5 = 0.2
    assert math.isclose(m.precision_at_k, 0.2, rel_tol=1e-4)
    # Recall@5: 1 / 2 = 0.5
    assert math.isclose(m.recall_at_k, 0.5, rel_tol=1e-4)
    # MRR: rank 1 -> 1.0
    assert math.isclose(m.mrr, 1.0, rel_tol=1e-4)
    # IDCG based on min(5, 2 expected) = 2 items
    expected_ndcg = 1.0 / (1.0 + 1.0 / math.log2(3))
    assert math.isclose(m.ndcg_at_k, expected_ndcg, rel_tol=1e-4)


def test_metrics_single_expected_file_at_rank_two():
    """Test a single expected file appearing at rank 2 in unique file list."""
    expected = ["src/target.py"]
    chunks = [
        make_chunk("src/wrong.py", 0.9),
        make_chunk("src/target.py", 0.8),
    ]

    m = compute_retrieval_metrics(chunks, expected, k=3)
    assert math.isclose(m.precision_at_k, 1 / 3, rel_tol=1e-4)
    assert math.isclose(m.recall_at_k, 1.0, rel_tol=1e-4)
    # First match at rank 2 -> MRR = 0.5
    assert math.isclose(m.mrr, 0.5, rel_tol=1e-4)
    # DCG = 1/log2(3) = 0.63092975; IDCG = 1/log2(2) = 1.0
    assert math.isclose(m.ndcg_at_k, 1.0 / math.log2(3), rel_tol=1e-4)


def test_metrics_empty_expected_files_raises():
    """Empty expected_files must raise ValueError."""
    with pytest.raises(ValueError, match="expected_files cannot be empty"):
        compute_retrieval_metrics([make_chunk("a.py")], expected_files=[], k=5)


def test_source_type_classification():
    """Verify source-type classification for code, docs, and tests."""
    assert classify_source_type("src/click/core.py") == "code"
    assert classify_source_type("click/parser.py") == "code"
    assert classify_source_type("docs/index.md") == "docs"
    assert classify_source_type("README.md") == "docs"
    assert classify_source_type("tests/test_basic.py") == "tests"
    assert classify_source_type("test_core.py") == "tests"

    chunks = [
        make_chunk("src/a.py"),
        make_chunk("docs/guide.md"),
        make_chunk("tests/test_a.py"),
    ]
    bd = compute_source_breakdown(chunks, k=3)
    assert math.isclose(bd.code_share, 1 / 3, rel_tol=1e-4)
    assert math.isclose(bd.docs_share, 1 / 3, rel_tol=1e-4)
    assert math.isclose(bd.tests_share, 1 / 3, rel_tol=1e-4)


# =========================================================================
# 2. Golden Dataset Loader & Validation Error Tests
# =========================================================================


def test_dataset_line_numbered_syntax_error(tmp_path: Path):
    """Corrupt JSON must fail with an error explicitly naming the line number."""
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text(
        '{"id": "q1", "repo": "r", "repo_commit": "c", "question": "q", "category": "code_understanding"}\n'
        'NOT VALID JSON ON LINE 2\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 2"):
        load_golden_dataset(bad_file)


def test_dataset_line_numbered_schema_error(tmp_path: Path):
    """Schema violation must fail with an error explicitly naming the line number."""
    bad_file = tmp_path / "bad_schema.jsonl"
    bad_file.write_text(
        '{"id": "q1", "repo": "r", "repo_commit": "c", "question": "q", "category": "code_understanding"}\n'
        '{"id": "q2", "repo": "r", "repo_commit": "c", "question": "q", "category": "INVALID_CATEGORY"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 2"):
        load_golden_dataset(bad_file)


def test_dataset_load_and_save_roundtrip(tmp_path: Path):
    """Dataset serialization and loading must preserve all fields including answerable."""
    q = GoldenQuestion(
        id="test-1",
        repo="myrepo",
        repo_commit="sha-1234",
        question="Where is main?",
        category="code_understanding",
        expected_files=["src/main.py"],
        expected_answer_notes="In main.py",
        expected_commits=["sha-1234"],
        answerable=True,
    )

    save_path = tmp_path / "test.jsonl"
    save_golden_dataset([q], save_path)
    loaded = load_golden_dataset(save_path)

    assert len(loaded) == 1
    assert loaded[0] == q


def test_sample_dataset_valid():
    """Verify eval_samples/golden_sample.jsonl loads cleanly and validates."""
    sample_path = Path("eval_samples/golden_sample.jsonl")
    assert sample_path.is_file()
    questions = load_golden_dataset(sample_path)
    assert len(questions) == 6
    # 5 answerable, 1 unanswerable
    assert sum(1 for q in questions if q.answerable) == 5
    assert sum(1 for q in questions if not q.answerable) == 1


# =========================================================================
# 3. Evaluation Runner End-to-End on Fixture Repo
# =========================================================================


@pytest.fixture
def eval_fixture_repo(tmp_path: Path) -> tuple[Path, str]:
    """Create a temporary git repository with two source files."""
    repo_dir = tmp_path / "eval_repo"
    repo_dir.mkdir(parents=True)
    repo = git.Repo.init(repo_dir)

    src_dir = repo_dir / "src"
    src_dir.mkdir()
    (src_dir / "app.py").write_text("def run(): pass\n", encoding="utf-8")

    docs_dir = repo_dir / "docs"
    docs_dir.mkdir()
    (docs_dir / "guide.md").write_text("# Guide\nHow to run app.\n", encoding="utf-8")

    repo.index.add(["src/app.py", "docs/guide.md"])
    commit = repo.index.commit("Initial commit")
    return repo_dir, commit.hexsha


def test_runner_end_to_end_and_commit_mismatch(
    eval_fixture_repo: tuple[Path, str], tmp_path: Path
):
    """Verify runner end-to-end creates results.json and summary.md, and rejects commit mismatch."""
    repo_dir, commit_sha = eval_fixture_repo

    settings = Settings(
        app=AppSettings(),
        paths=PathsSettings(
            data_dir=tmp_path / "data",
            cache_dir=tmp_path / ".cache",
            llm_cache_dir=tmp_path / ".cache" / "llm",
        ),
        budgets=BudgetSettings(
            max_llm_calls=10, max_input_tokens=10000, max_output_tokens=10000
        ),
        llm=LLMSettings(
            tiers={
                "strong": LLMTierSettings(
                    provider="fake",
                    model="fake",
                    temperature=0.0,
                    max_output_tokens=64,
                )
            }
        ),
        ingestion=IngestionSettings(chunk_size=500, chunk_overlap=50),
        embedding=EmbeddingSettings(model="BAAI/bge-small-en-v1.5", dimensions=384),
        vector_store=VectorStoreSettings(
            mode="local",
            path=tmp_path / "qdrant",
            collection_prefix="repopilot_test",
        ),
        env=Environment(_env_file=None),
    )

    store = QdrantVectorStore(mode="local", path=tmp_path / "qdrant")
    embedder = FakeEmbedder(dimensions=384)

    try:
        # 1. Index the fixture repo
        index_repository(
            url=str(repo_dir),
            name="evalrepo",
            settings=settings,
            embedder=embedder,
            store=store,
        )

        # 2. Create golden dataset matching the fixture repo
        dataset_file = tmp_path / "golden.jsonl"
        q1 = GoldenQuestion(
            id="q-001",
            repo="evalrepo",
            repo_commit=commit_sha,
            question="Where is run() implemented?",
            category="code_understanding",
            expected_files=["src/app.py"],
            answerable=True,
        )
        q2_unanswerable = GoldenQuestion(
            id="q-002",
            repo="evalrepo",
            repo_commit=commit_sha,
            question="What is the database password?",
            category="debugging",
            expected_files=[],
            answerable=False,
        )
        save_golden_dataset([q1, q2_unanswerable], dataset_file)

        # 3. Run evaluation
        summary = run_retrieval_evaluation(
            dataset_path=dataset_file,
            retriever_name="dense",
            store=store,
            embedder=embedder,
            settings=settings,
            k=2,
            results_base_dir=tmp_path / "results",
        )

        assert summary.total_questions == 2
        assert summary.evaluated_questions == 1
        assert summary.skipped_unanswerable == 1
        assert summary.mean_recall_at_k >= 0.0

        # Check results files
        results_json = summary.output_dir / "results.json"
        summary_md = summary.output_dir / "summary.md"
        assert results_json.is_file()
        assert summary_md.is_file()

        data = json.loads(results_json.read_text(encoding="utf-8"))
        assert data["metadata"]["evaluated_questions"] == 1
        assert data["metadata"]["skipped_unanswerable"] == 1
        assert "source_type_distribution" in data

        md_text = summary_md.read_text(encoding="utf-8")
        assert "Retrieval Evaluation Summary" in md_text
        assert "Overall Metrics" in md_text
        assert "Source-Type Distribution" in md_text

        # 4. Verify commit mismatch failure
        mismatch_file = tmp_path / "mismatch.jsonl"
        bad_q = GoldenQuestion(
            id="q-bad",
            repo="evalrepo",
            repo_commit="wrong_commit_sha_12345",
            question="Where is run()?",
            category="code_understanding",
            expected_files=["src/app.py"],
            answerable=True,
        )
        save_golden_dataset([bad_q], mismatch_file)

        with pytest.raises(ValueError, match="Commit mismatch"):
            run_retrieval_evaluation(
                dataset_path=mismatch_file,
                retriever_name="dense",
                store=store,
                embedder=embedder,
                settings=settings,
                k=2,
            )

    finally:
        store.close()


# =========================================================================
# 4. Question Drafter with Fake LLM Provider
# =========================================================================


def test_drafter_writes_reviewed_false(
    eval_fixture_repo: tuple[Path, str], tmp_path: Path
):
    """Verify drafter generates candidates with reviewed=false and hints-only notes."""
    from scripts.draft_questions import CandidateDraft, draft_questions_from_repo

    repo_dir, commit_sha = eval_fixture_repo

    settings = Settings(
        app=AppSettings(),
        paths=PathsSettings(
            data_dir=tmp_path / "data",
            cache_dir=tmp_path / ".cache",
            llm_cache_dir=tmp_path / ".cache" / "llm",
        ),
        budgets=BudgetSettings(
            max_llm_calls=10, max_input_tokens=10000, max_output_tokens=10000
        ),
        llm=LLMSettings(
            tiers={
                "strong": LLMTierSettings(
                    provider="fake",
                    model="fake",
                    temperature=0.0,
                    max_output_tokens=64,
                )
            }
        ),
        ingestion=IngestionSettings(chunk_size=500, chunk_overlap=50),
        embedding=EmbeddingSettings(model="BAAI/bge-small-en-v1.5", dimensions=384),
        vector_store=VectorStoreSettings(
            mode="local",
            path=tmp_path / "qdrant",
            collection_prefix="repopilot_test",
        ),
        env=Environment(_env_file=None),
    )

    store = QdrantVectorStore(mode="local", path=tmp_path / "qdrant")
    embedder = FakeEmbedder(dimensions=384)

    try:
        index_repository(
            url=str(repo_dir),
            name="draftrepo",
            settings=settings,
            embedder=embedder,
            store=store,
        )

        mock_llm = MagicMock()
        mock_llm.generate_structured.return_value = CandidateDraft(
            question="How does run() work in this repository?",
            category="code_understanding",
            hint_files=["src/app.py"],
            answer_notes="run() is a stub function.",
        )

        out_file = tmp_path / "candidates.jsonl"
        draft_questions_from_repo(
            repo_slug="draftrepo",
            n=2,
            local=True,
            output_path=out_file,
            store=store,
            llm=mock_llm,
            settings=settings,
        )

        assert out_file.is_file()
        lines = [json.loads(l) for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2
        for record in lines:
            assert record["reviewed"] is False
            assert record["answerable"] is True
            assert "HINTS ONLY" in record["expected_answer_notes"]
            assert record["source_seed_file"] in ["src/app.py", "docs/guide.md"]
            assert record["source_seed_file"] in record["expected_files"]
            assert "src/app.py" in record["expected_files"]

    finally:
        store.close()
