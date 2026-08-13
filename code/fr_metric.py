"""
fr_metric.py — Flip Rate (FR) measurement from retrieval logs (post-hoc, no GPU).

Implements the diagnostic metrics of the main paper:
  * FR        (Eq. 2): fraction of queries whose fused top-1 differs from the
                       measurement-anchor-only top-1.
  * FR@k(pos): fraction of queries whose rank-k document comes from the
               secondary source.
  * FR@k(set): fraction of queries with at least one secondary-source document
               within the top-k.

Input format — one run directory per condition, each containing a
`retrieval.jsonl` with one JSON record per query:

  {"query_id": "...",
   "docs": [{"rank": 1, "doc_id": "...", "source_name": "wiki_full", ...},
            {"rank": 2, "doc_id": "...", "source_name": "medrag_pubmed", ...}]}

Usage:
  python fr_metric.py \
      --anchor-dir  runs/mixed_anchor_only_seed42 \
      --multi-dirs  runs/mixed_dense_rrf_seed42 runs/mixed_hybrid_seed42 \
      --labels      CH-D CH-H \
      --secondary-source medrag_pubmed \
      --save-csv    frk_results.csv
"""

import os
import json
import argparse
import csv
from collections import defaultdict


# ------------------------------------------------------------
# retrieval.jsonl loading
# ------------------------------------------------------------

def load_retrieval(run_dir: str) -> dict:
    """Read retrieval.jsonl and return {query_id: [doc_dict, ...]} sorted by rank."""
    path = os.path.join(run_dir, "retrieval.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"retrieval.jsonl not found: {path}")

    data = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = rec.get("query_id") or rec.get("qid")
            docs = rec.get("docs", [])
            if not qid or not docs:
                continue
            data[qid] = sorted(docs, key=lambda d: d.get("rank", 999))
    return data


def _source_of(doc: dict) -> str:
    return doc.get("source_name") or doc.get("source_leg") or doc.get("leg") or ""


# ------------------------------------------------------------
# FR (Eq. 2): anchor-only top-1 vs fused top-1
# ------------------------------------------------------------

def compute_fr(anchor_data: dict, multi_data: dict) -> float:
    """FR = (1/|Q|) * sum_q  1[ d_M*(q) != d_1*(q) ]  over common queries."""
    common = set(anchor_data.keys()) & set(multi_data.keys())
    if not common:
        return None
    flips = 0
    for qid in common:
        a_top1 = anchor_data[qid][0].get("doc_id") if anchor_data[qid] else None
        m_top1 = multi_data[qid][0].get("doc_id") if multi_data[qid] else None
        if a_top1 != m_top1:
            flips += 1
    return flips / len(common)


# ------------------------------------------------------------
# FR@k positional distribution
# ------------------------------------------------------------

def compute_frk_pos(multi_data: dict, secondary_source: str, k_max: int = 5):
    """FR@k(pos): fraction of queries whose rank-k document is from the secondary source."""
    counts = defaultdict(int)
    total = 0
    for _, docs in multi_data.items():
        total += 1
        for k in range(1, k_max + 1):
            if k - 1 < len(docs) and _source_of(docs[k - 1]) == secondary_source:
                counts[k] += 1
    return {k: (counts[k] / total if total else 0.0) for k in range(1, k_max + 1)}, total


def compute_frk_set(multi_data: dict, secondary_source: str, k_max: int = 5):
    """FR@k(set): fraction of queries with >=1 secondary-source document in the top-k."""
    counts = defaultdict(int)
    total = 0
    for _, docs in multi_data.items():
        total += 1
        found = False
        for k in range(1, k_max + 1):
            if not found and k - 1 < len(docs) \
                    and _source_of(docs[k - 1]) == secondary_source:
                found = True
            if found:
                counts[k] += 1
    return {k: (counts[k] / total if total else 0.0) for k in range(1, k_max + 1)}, total


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FR / FR@k measurement from retrieval.jsonl logs")
    parser.add_argument("--anchor-dir", type=str, default=None,
                        help="anchor-only (single-source) run directory, enables FR (Eq. 2)")
    parser.add_argument("--multi-dirs", type=str, nargs="+", required=True,
                        help="multi-source run directories to evaluate")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="labels for --multi-dirs (e.g., CH-D CH-H)")
    parser.add_argument("--secondary-source", type=str, default="medrag_pubmed")
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--save-csv", type=str, default=None)
    args = parser.parse_args()

    labels = args.labels if args.labels else [os.path.basename(d) for d in args.multi_dirs]

    anchor_data = None
    if args.anchor_dir:
        anchor_data = load_retrieval(args.anchor_dir)
        print(f"[anchor] {args.anchor_dir}  ({len(anchor_data)} queries)")

    csv_rows = []
    for mdir, label in zip(args.multi_dirs, labels):
        multi_data = load_retrieval(mdir)
        print(f"\n[{label}] {mdir}  ({len(multi_data)} queries)")

        if anchor_data:
            fr = compute_fr(anchor_data, multi_data)
            print(f"  FR (Eq. 2, top-1 flip vs anchor-only): {fr:.4f}")

        frk_pos, n = compute_frk_pos(multi_data, args.secondary_source, args.k_max)
        frk_set, _ = compute_frk_set(multi_data, args.secondary_source, args.k_max)

        print(f"  {'k':>3}  {'FR@k(pos)':>10}  {'FR@k(set)':>10}")
        for k in range(1, args.k_max + 1):
            print(f"  {k:>3}  {frk_pos[k]:>10.4f}  {frk_set[k]:>10.4f}")

        for k in range(1, args.k_max + 1):
            csv_rows.append({
                "label": label, "run_dir": mdir, "k": k,
                "frk_pos": round(frk_pos[k], 4), "frk_set": round(frk_set[k], 4),
                "n_queries": n, "secondary_source": args.secondary_source,
            })

    if args.save_csv and csv_rows:
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n[saved] {args.save_csv}")


if __name__ == "__main__":
    main()
