# Supplementary Material — "When Multi-Source RAG Helps—and When It Doesn't: A Pre-Deployment Diagnostic via Label-Free Hybrid Source Routing"

Anonymous supplementary repository for a WSDM 2027 submission.

## Contents

- `supplementary.pdf` — Supplementary appendices **A–Q**, referenced from the main paper as `Supp. App. X`.
  Table/figure numbers carry an `S` prefix; unprefixed numbers (e.g., Table 1, Eq. 2, §5.1) refer to the main paper.

| App | Contents |
|---|---|
| A | Corpus quality control protocol (corpus statistics) |
| B | Four-LLM replication, closed-book diagnosis, 70B scale check |
| C | Query-ratio generalization |
| D | Retriever generalization: Contriever / Contriever-MSMARCO |
| E | Retriever generalization: BGE-large-en-v1.5 |
| F | Statistical significance tables; HF-RAG score-normalization robustness |
| G | Domain generalization (Legal/Financial), CP/JSD proxies, embedding geometry |
| H | Error analysis |
| I | Extended retrieval quality (Recall@k, MRR), FR validity, MedCPT re-ranking |
| J | Per-type EM analysis |
| K | Condition summary and experimental design matrix |
| L | End-to-end conditioned Hybrid pilot |
| M | Anchor symmetry, Budget-U robustness |
| N | Supervised routing baseline (LogReg, N=200) |
| O | Candidate-level BM25 routing: per-query evidence, tie-break control |
| P | z-CombSUM score-level diagnosis |
| Q | Reproducibility details (prompts, hyperparameters, splits, compute) |

Note: tables promoted into the main paper (four-LLM generalization, encoder-family FR, retrieval quality) are retained here for completeness. The JSD Layer-2 screening table appears here only; the main paper reports its four values inline.

## Code

`code/` contains reference implementations of the paper's core measurement and fusion logic, faithful to the experiment pipeline:

- **`code/fr_metric.py`** — Flip Rate measurement from retrieval logs (no GPU):
  FR (Eq. 2, fused top-1 vs. anchor-only top-1), FR@k(pos), and FR@k(set).
  Operates on `retrieval.jsonl` run logs; the record schema is documented in the file header.
- **`code/hybrid_fusion.py`** — the two fusion conditions compared in the paper:
  per-source Dense RRF (`rrf_fuse`, the tie-lock baseline of §3.2) and candidate-level
  Hybrid (`hybrid_fuse`, the two-stage RRF of §3.3 with query-time BM25 over the joint pool;
  `rank_bm25` BM25Okapi, defaults k1=1.5, b=0.75). Running the file directly
  (`python hybrid_fusion.py`) executes a self-contained toy demonstration: both sources'
  rank-1 candidates receive the identical RRF score 1/61, the stable sort hands top-1 to the
  first-listed source (FR=0), and candidate-level Hybrid routes top-1 to the lexically
  matching source.

Dependency: `pip install rank_bm25`.

## Data

Full code (retrieval/generation pipeline, prompt templates, index settings, dataset splits) and per-seed outputs will be released upon publication (see main paper, Ethical Considerations — Reproducibility).

All corpora and benchmarks are publicly available under licenses permitting research use (Wikipedia/Wikidata, PubMed, MedRAG suite, CUAD, FiQA-2018, TAT-QA).
