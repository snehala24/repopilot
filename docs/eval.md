# Retrieval Evaluation Framework & V1 Baseline

This document describes RepoPilot's evaluation methodology, metric definitions, dataset schema, curation guidelines, and the initial V1 baseline evaluation results.

---

## 1. Evaluation Methodology

RepoPilot's retrieval evaluation is designed specifically for codebases:
1. **Developer-Grade Questions**: Realistic developer queries across five distinct categories (`code_understanding`, `debugging`, `change_impact`, `regression`, `history`).
2. **Commit-Pinned Datasets**: Every question is pinned to the exact repository commit SHA indexed in the vector database. The evaluation runner refuses to evaluate mismatched commits to prevent drifting benchmarks.
3. **Pure Python, Deterministic Metrics**: Metrics require no LLM in the loop, ensuring 100% reproducible, zero-cost scoring across experiments.
4. **Structured Artifacts**: Every evaluation run exports:
   - `results.json`: Full machine-readable breakdown, query metrics, retrieved chunks, and aggregated scores.
   - `summary.md`: Human-readable summary table, source-type distribution, category breakdowns, and the 5 worst-performing queries for rapid qualitative inspection.

---

## 2. Metric Definitions

Relevance is evaluated at the **file level**: a retrieved chunk is relevant if its repo-relative file path matches one of the question's `expected_files`.

### Unique File Conversion
For file-level ranking metrics (Recall@K, MRR, NDCG@K), the ranked list of retrieved chunks is converted to a ranked list of unique file paths, preserving the rank of each file's first occurrence. Duplicate chunks from the same file are collapsed.

### Precision@K (Chunk-Level)
Measures the proportion of retrieved **chunks** in the top-K that originate from relevant files:
$$\text{Precision@K} = \frac{|\{c \in \text{chunks}_{1..K} \mid c.\text{file\_path} \in \text{expected\_files}\}|}{K}$$

### Recall@K (Unique File-Level)
Measures the fraction of expected relevant files discovered in the top-K unique files:
$$\text{Recall@K} = \frac{|\{f \in \text{files}_{1..K} \mid f \in \text{expected\_files}\}|}{|\text{expected\_files}|}$$

### MRR (Mean Reciprocal Rank, Unique File-Level)
Measures how high the first relevant file appears in the ranked list of unique files:
$$\text{MRR} = \begin{cases} \frac{1}{\text{rank}}, & \text{if a relevant file is found in top-}K \\ 0.0, & \text{otherwise} \end{cases}$$

### NDCG@K (Normalized Discounted Cumulative Gain, Unique File-Level)
Measures the ranking quality of unique files using binary relevance:
$$\text{DCG@K} = \sum_{i=1}^{K} \frac{\text{rel}_i}{\log_2(i + 1)}$$
$$\text{IDCG@K} = \sum_{i=1}^{\min(K, |\text{expected\_files}|)} \frac{1}{\log_2(i + 1)}$$
$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$
where $\text{rel}_i \in \{0, 1\}$. When $\text{IDCG@K} = 0$, $\text{NDCG@K} = 0.0$.

### Source-Type Classification Rule
To analyze what types of repository content the retriever favors, each retrieved chunk is classified into one of three buckets based on its file path:
- **`tests`**: Path starts with `tests/`, ends with `_test.py`, or file name starts with `test_`.
- **`docs`**: Path starts with `docs/`, has extension `.md` or `.rst`, or file name starts with `readme` (case-insensitive).
- **`code`**: All other source files (e.g. `src/`, package implementation code).

The evaluation runner computes and reports the average percentage share of each source type across all top-K retrieved chunks.

---

## 3. Golden Dataset Schema & Curation Guidelines

### Pydantic Model (`GoldenQuestion`)
Golden questions are stored in JSONL format, validated with `repopilot.evaluation.dataset.GoldenQuestion`:

```python
class GoldenQuestion(BaseModel):
    id: str                          # Unique identifier (e.g. click-001)
    repo: str                        # Repository slug (e.g. click)
    repo_commit: str                 # Indexed commit SHA
    question: str                    # Developer query
    category: Literal[               # Question category
        "code_understanding",
        "debugging",
        "change_impact",
        "regression",
        "history",
    ]
    expected_files: list[str]        # Expected repo-relative POSIX paths
    expected_answer_notes: str = ""  # Human explanation of the expected answer
    expected_commits: list[str] | None = None
    answerable: bool = True          # False for out-of-scope or unanswerable queries
```

### Unanswerable Questions (`answerable = False`)
When `answerable: false`, the query cannot or should not be answered from the codebase (e.g. out-of-scope, missing features, secret credentials).
- **Retrieval Evaluation**: Unanswerable questions are safely skipped during retrieval metric computation. The runner logs the skipped count.
- **Future Generation Evaluation**: Unanswerable questions will be evaluated to verify that the LLM appropriately refuses to answer instead of hallucinating.

### Candidate Drafting Protocol (`scripts/draft_questions.py`)
To rapidly bootstrap golden datasets without introducing hallucinated ground truth:
1. **Stratified Sampling**: Chunks are sampled from Qdrant in roughly equal proportions across code, docs, and tests.
2. **AI Question Generation**: The `strong` LLM tier drafts realistic questions based on sampled chunks.
3. **Mandatory Human Verification**:
   - Every candidate is written with `reviewed: false`.
   - The candidate's `expected_files` are marked as **HINTS ONLY**.
   - A human curator must verify that the expected files actually contain the answer and that the question represents genuine developer intent before promoting the question to a golden dataset.

---

## 4. V1 Baseline Retrieval Results (pallets/click)

The following baseline was produced by running `DenseRetriever` with FastEmbed (`BAAI/bge-small-en-v1.5`, 384-dim) against `pallets/click` (commit `2247b35ea1c47c727d7a06e51fa280e12a863ff6`, 1,732 chunks in Qdrant) using `eval_samples/golden_sample.jsonl`:

### Overall Metrics ($K=5$)

| Metric | V1 Dense Baseline |
| :--- | :--- |
| **Precision@5** | **0.0800** |
| **Recall@5** | **0.2000** |
| **MRR** | **0.0667** |
| **NDCG@5** | **0.1000** |
| **Evaluated Questions** | 5 (1 skipped unanswerable) |

### Source-Type Distribution (Top-5 Chunks)

| Source Type | Share |
| :--- | :--- |
| **Docs** (`docs/`, markdown) | **64.0%** |
| **Tests** (`tests/`) | **24.0%** |
| **Code** (`src/` implementation) | **12.0%** |

### Breakdown by Category

| Category | Count | Precision@5 | Recall@5 | MRR | NDCG@5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `code_understanding` | 2 | 0.2000 | 0.5000 | 0.1667 | 0.2500 |
| `debugging` | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `change_impact` | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `regression` | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Failure Analysis & Insights
1. **Prose vs. Code Bias**: Pure dense embedding models strongly favor natural language prose. Documentation files (`docs/complex.md`, `docs/commands-and-groups.md`, `changes.md`) matched queries far more strongly than actual implementation files (`src/click/core.py`, `src/click/parser.py`), capturing 64% of top-5 retrieval slots.
2. **Exact Symbol Matching Gap**: Questions asking for specific classes (e.g. `BaseCommand`, `OptionParser`) failed in dense search because dense semantic vectors do not reward exact identifier matches.
3. **Motivation for Advanced RAG**: These empirical results establish a clear baseline and demonstrate why subsequent milestones (sparse BM25 search, Reciprocal Rank Fusion hybrid retrieval, cross-encoder reranking, and Tree-sitter AST-aware chunking) will dramatically improve code retrieval.

---

## 5. CLI Usage

### Running Retrieval Evaluation
```powershell
# Run evaluation on Docker Qdrant (server mode)
python scripts\run_retrieval_eval.py --retriever dense --dataset eval_samples/golden_sample.jsonl --k 5 --server

# Run evaluation on local embedded storage
python scripts\run_retrieval_eval.py --retriever dense --dataset eval_samples/golden_sample.jsonl --k 5 --local
```

### Drafting Candidate Questions
```powershell
# Draft 20 candidate questions from an indexed repository (requires Gemini API key)
python scripts\draft_questions.py --repo click --n 20 --server
```
Outputs candidate questions to `data/eval/candidates_<slug>.jsonl` with `reviewed: false` for human review.
