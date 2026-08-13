"""
hybrid_fusion.py — Reference implementation of the two fusion conditions
compared in the main paper (faithful to the experiment pipeline):

  * rrf_fuse    — per-source Dense RRF (the pathological baseline, Sec. 3.2):
                  with disjoint corpora, every source's rank-1 receives the
                  identical score 1/(rrf_k+1), and Python's stable sort resolves
                  the tie by insertion order = source-list order, so the fused
                  top-1 goes to the first-listed (anchor) source on every query
                  (FR = 0).
  * hybrid_fuse — candidate-level Hybrid (Sec. 3.3): a second RRF term from
                  query-time BM25 computed over the *joint* candidate pool.
                  BM25 ranks all K*k candidates globally (no per-source
                  partitioning, no ties), breaking the lock-in and aligning
                  top-1 with the query's lexical profile.

Dependencies: rank_bm25 (BM25Okapi, defaults k1=1.5, b=0.75 as in the paper).
Run `python hybrid_fusion.py` for a self-contained toy demonstration of the
tie-lock and its escape (no index or GPU required).
"""

import re
import string
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

RRF_K = 60  # RRF constant; every source's rank-1 scores 1/(60+1)


@dataclass
class RetrievedDoc:
    doc_id: str
    content: str
    source_name: str
    rank_in_source: int      # 1-based rank within its own source's ranked list
    fused_score: float = 0.0
    bm25_score: float = 0.0
    final_rank: int = 0


# ------------------------------------------------------------
# BM25 over the joint candidate pool (candidate-level: only documents
# retrieved by Dense are rescored; no full-corpus BM25 index is used)
# ------------------------------------------------------------

def tokenize(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    return text.split()


def bm25_score_docs(query: str, docs: list) -> dict:
    """BM25 scores for the joint candidate pool, {doc_id: score}."""
    if not docs:
        return {}
    tokenized_corpus = [tokenize(d.content or "") for d in docs]
    tokenized_query = tokenize(query)
    if not any(tokenized_corpus) or not tokenized_query:
        return {d.doc_id: 0.0 for d in docs}
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    return {doc.doc_id: float(score) for doc, score in zip(docs, scores)}


# ------------------------------------------------------------
# Fusion conditions
# ------------------------------------------------------------

def rrf_fuse(source_results: dict, top_n: int, rrf_k: int = RRF_K) -> list:
    """Per-source Dense RRF (baseline).

    source_results: {source_name: [RetrievedDoc, ...]} — one independently
    retrieved ranked list per source, in source-list order.

    NOTE (the pathology, Sec. 3.2): with disjoint corpora every source's
    rank-1 gets the identical score 1/(rrf_k+1). `sorted` is stable, so the
    tie is resolved by dict insertion order, i.e., by the order sources are
    listed — the first-listed source wins the fused top-1 on every query.
    The public MedRAG implementation exhibits the same behavior (see the
    main paper, footnote in Sec. 3.2).
    """
    rrf_scores: dict = {}
    doc_store: dict = {}
    for _, docs in source_results.items():
        for doc in docs:
            score = 1.0 / (rrf_k + doc.rank_in_source)
            rrf_scores[doc.doc_id] = rrf_scores.get(doc.doc_id, 0.0) + score
            if doc.doc_id not in doc_store:
                doc_store[doc.doc_id] = doc
    sorted_ids = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)
    final = []
    for rank, did in enumerate(sorted_ids[:top_n], start=1):
        doc = doc_store[did]
        doc.fused_score = rrf_scores[did]
        doc.final_rank = rank
        final.append(doc)
    return final


def hybrid_fuse(query: str, source_results: dict, top_n: int,
                rrf_k: int = RRF_K) -> list:
    """Candidate-level Hybrid (two-stage RRF, Sec. 3.3).

    score_hybrid(d) = 1/(rrf_k + rank_DenseRRF(d)) + 1/(rrf_k + rank_BM25(d))

    where rank_BM25 ranks the joint Dense-retrieved pool globally as a single
    list — no per-source partitioning, hence no structural ties.
    """
    doc_store: dict = {}
    for _, docs in source_results.items():
        for doc in docs:
            if doc.doc_id not in doc_store:
                doc_store[doc.doc_id] = doc
    all_docs = list(doc_store.values())
    if not all_docs:
        return []

    # First stage: per-source Dense RRF component.
    dense_rrf: dict = {}
    for _, docs in source_results.items():
        for doc in docs:
            score = 1.0 / (rrf_k + doc.rank_in_source)
            dense_rrf[doc.doc_id] = dense_rrf.get(doc.doc_id, 0.0) + score

    # Second stage: BM25 rank over the joint pool, converted to RRF.
    bm25_scores = bm25_score_docs(query, all_docs)
    sorted_by_bm25 = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
    bm25_rrf: dict = {}
    for bm25_rank, (doc_id, _) in enumerate(sorted_by_bm25, start=1):
        bm25_rrf[doc_id] = 1.0 / (rrf_k + bm25_rank)

    hybrid_scores = {doc_id: dense_rrf.get(doc_id, 0.0) + bm25_rrf.get(doc_id, 0.0)
                     for doc_id in doc_store}

    sorted_ids = sorted(hybrid_scores, key=lambda d: hybrid_scores[d], reverse=True)
    final = []
    for rank, did in enumerate(sorted_ids[:top_n], start=1):
        doc = doc_store[did]
        doc.fused_score = hybrid_scores[did]
        doc.bm25_score = bm25_scores.get(did, 0.0)
        doc.final_rank = rank
        final.append(doc)
    return final


# ------------------------------------------------------------
# Toy demonstration: tie-lock (FR=0) and the Hybrid escape
# ------------------------------------------------------------

if __name__ == "__main__":
    # Two disjoint sources, anchor listed first. The query is biomedical.
    query = "does aspirin reduce the risk of myocardial infarction"

    anchor_docs = [
        RetrievedDoc("wiki:1", "Aspirin is a medication used to reduce pain, "
                               "fever, or inflammation.", "wiki_full", 1),
        RetrievedDoc("wiki:2", "The history of aspirin begins with willow bark "
                               "remedies in antiquity.", "wiki_full", 2),
    ]
    secondary_docs = [
        RetrievedDoc("pubmed:1", "Randomized trial: aspirin reduces the risk of "
                                 "myocardial infarction in men over 50.",
                     "medrag_pubmed", 1),
        RetrievedDoc("pubmed:2", "Meta-analysis of antiplatelet therapy and "
                                 "cardiovascular outcomes.", "medrag_pubmed", 2),
    ]
    source_results = {"wiki_full": anchor_docs, "medrag_pubmed": secondary_docs}

    dense = rrf_fuse(source_results, top_n=4)
    print("Per-source Dense RRF (baseline):")
    for d in dense:
        print(f"  [{d.final_rank}] {d.source_name:14s} {d.doc_id:9s} "
              f"score={d.fused_score:.6f}")
    print("  -> both rank-1 docs score identically "
          f"(1/{RRF_K + 1} = {1.0 / (RRF_K + 1):.6f}); the stable sort hands "
          "top-1 to the first-listed source (tie-lock, FR=0).\n")

    hybrid = hybrid_fuse(query, source_results, top_n=4)
    print("Candidate-level Hybrid:")
    for d in hybrid:
        print(f"  [{d.final_rank}] {d.source_name:14s} {d.doc_id:9s} "
              f"score={d.fused_score:.6f} bm25={d.bm25_score:.3f}")
    print("  -> BM25 over the joint pool has no ties and routes top-1 to the "
          "lexically matching source (the escape; FR>0).")
