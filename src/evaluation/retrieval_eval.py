"""
Retrieval evaluation module: Recall@K, MRR, NDCG, MAP.

Evaluates the RetrieverAgent on PubMedQA labeled set, where each question
has known relevant PMIDs from the associated context passages.

Metrics:
  - Recall@K:  fraction of relevant docs found in top-K retrieved
  - MRR:       mean reciprocal rank of first relevant doc
  - NDCG@K:    normalized discounted cumulative gain (position-weighted)
  - MAP:       mean average precision across all queries
  - Hit Rate:  binary — did any relevant doc appear in top-K?

Usage:
    python -m src.evaluation.retrieval_eval --top-k 10 --max-samples 200
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
from loguru import logger
from tqdm import tqdm

from src.config import settings


@dataclass
class RetrievalMetrics:
    """Aggregate retrieval metrics for a set of queries."""
    recall_at_k: dict[int, float] = field(default_factory=dict)  # k → recall
    mrr: float = 0.0
    ndcg_at_k: dict[int, float] = field(default_factory=dict)   # k → NDCG
    map_score: float = 0.0
    hit_rate_at_k: dict[int, float] = field(default_factory=dict)
    n_queries: int = 0
    avg_relevant_per_query: float = 0.0

    def summary(self) -> dict:
        return {
            "mrr": round(self.mrr, 4),
            "map": round(self.map_score, 4),
            "recall@5": round(self.recall_at_k.get(5, 0), 4),
            "recall@10": round(self.recall_at_k.get(10, 0), 4),
            "ndcg@5": round(self.ndcg_at_k.get(5, 0), 4),
            "ndcg@10": round(self.ndcg_at_k.get(10, 0), 4),
            "hit_rate@5": round(self.hit_rate_at_k.get(5, 0), 4),
            "hit_rate@10": round(self.hit_rate_at_k.get(10, 0), 4),
            "n_queries": self.n_queries,
        }

    def print_table(self) -> None:
        s = self.summary()
        logger.info(
            f"\n{'─'*50}\n"
            f"  Retrieval Evaluation Results\n"
            f"{'─'*50}\n"
            f"  Queries evaluated:    {s['n_queries']}\n"
            f"  MRR:                  {s['mrr']:.4f}\n"
            f"  MAP:                  {s['map']:.4f}\n"
            f"  Recall@5:             {s['recall@5']:.4f}\n"
            f"  Recall@10:            {s['recall@10']:.4f}\n"
            f"  NDCG@5:               {s['ndcg@5']:.4f}\n"
            f"  NDCG@10:              {s['ndcg@10']:.4f}\n"
            f"  Hit Rate@5:           {s['hit_rate@5']:.4f}\n"
            f"  Hit Rate@10:          {s['hit_rate@10']:.4f}\n"
            f"{'─'*50}"
        )


# ── Per-query metric functions ────────────────────────────────────────────────

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant docs found in top-k retrieved."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1/rank of the first relevant document. 0 if none found."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """
    NDCG@K: binary relevance (1 if relevant, 0 otherwise).
    DCG = sum(rel_i / log2(i+1)) for i in 1..k
    IDCG = DCG of ideal ranking (all relevant docs at top)
    """
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)

    # Ideal DCG: relevant docs ranked at positions 1, 2, ..., min(|relevant|, k)
    ideal_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_count + 1))

    return dcg / idcg if idcg > 0 else 0.0


def average_precision(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Average precision for a single query."""
    if not relevant_ids:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant_ids) if relevant_ids else 0.0


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1 if at least one relevant doc is in top-k, else 0."""
    return float(bool(set(retrieved_ids[:k]) & relevant_ids))


# ── PubMedQA gold relevance loader ────────────────────────────────────────────

def load_pubmedqa_relevance(
    path: Optional[Path] = None,
    max_samples: int = 500,
) -> list[dict]:
    """
    Load PubMedQA labeled set and extract (question, relevant_pubids) pairs.
    Each labeled entry has contexts with known relevant PMIDs.

    Returns list of {question, relevant_ids: set[str]}
    """
    path = path or settings.data_raw_dir / "pubmedqa" / "labeled_train.jsonl"
    if not path.exists():
        logger.warning(f"PubMedQA labeled file not found: {path}")
        return []

    pairs = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            pubid = str(record.get("pubid", ""))
            question = record.get("question", "")
            if not question or not pubid:
                continue
            # The pubid itself is the relevant document
            pairs.append({"question": question, "relevant_ids": {pubid}})
            if len(pairs) >= max_samples:
                break

    logger.info(f"Loaded {len(pairs)} PubMedQA relevance pairs from {path}")
    return pairs


# ── Main evaluation function ───────────────────────────────────────────────────

def evaluate_retriever(
    retriever=None,
    top_k_values: list[int] = [1, 3, 5, 10],
    max_samples: int = 200,
    pubmedqa_path: Optional[Path] = None,
    log_to_mlflow: bool = True,
) -> RetrievalMetrics:
    """
    Evaluate the RetrieverAgent on PubMedQA.

    For each question, retrieves top-K documents and computes
    Recall@K, MRR, NDCG@K, MAP against the gold relevant pubids.
    """
    from src.agents.retriever import RetrieverAgent

    ret = retriever or RetrieverAgent(top_k=max(top_k_values), use_mmr=False)
    pairs = load_pubmedqa_relevance(pubmedqa_path, max_samples=max_samples)

    if not pairs:
        logger.error("No evaluation pairs found. Run data download first.")
        return RetrievalMetrics()

    max_k = max(top_k_values)

    per_query_rr = []
    per_query_ap = []
    per_k_recalls: dict[int, list[float]] = {k: [] for k in top_k_values}
    per_k_ndcgs: dict[int, list[float]] = {k: [] for k in top_k_values}
    per_k_hits: dict[int, list[float]] = {k: [] for k in top_k_values}
    total_relevant = []

    for pair in tqdm(pairs, desc="Retrieval evaluation"):
        query = pair["question"]
        relevant = pair["relevant_ids"]
        total_relevant.append(len(relevant))

        docs = ret.retrieve(query, top_k=max_k, use_mmr=False)

        # Get retrieved pubids (use source field which encodes pubid)
        retrieved_ids = []
        for doc in docs:
            pid = doc.metadata.get("pubid", "")
            if not pid:
                # Try parsing from source string like "PubMedQA:12345678"
                src = doc.source
                if ":" in src:
                    pid = src.split(":")[-1]
            retrieved_ids.append(pid)

        per_query_rr.append(reciprocal_rank(retrieved_ids, relevant))
        per_query_ap.append(average_precision(retrieved_ids, relevant))

        for k in top_k_values:
            per_k_recalls[k].append(recall_at_k(retrieved_ids, relevant, k))
            per_k_ndcgs[k].append(ndcg_at_k(retrieved_ids, relevant, k))
            per_k_hits[k].append(hit_rate_at_k(retrieved_ids, relevant, k))

    metrics = RetrievalMetrics(
        mrr=float(np.mean(per_query_rr)),
        map_score=float(np.mean(per_query_ap)),
        recall_at_k={k: float(np.mean(v)) for k, v in per_k_recalls.items()},
        ndcg_at_k={k: float(np.mean(v)) for k, v in per_k_ndcgs.items()},
        hit_rate_at_k={k: float(np.mean(v)) for k, v in per_k_hits.items()},
        n_queries=len(pairs),
        avg_relevant_per_query=float(np.mean(total_relevant)),
    )

    metrics.print_table()

    if log_to_mlflow:
        try:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_name)
            with mlflow.start_run(run_name="retrieval_evaluation"):
                mlflow.log_metrics(metrics.summary())
                mlflow.log_param("max_samples", max_samples)
                mlflow.log_param("top_k_values", str(top_k_values))
                mlflow.log_param("embedding_model", settings.embedding_model)
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

    return metrics


# ── Compare MMR vs non-MMR retrieval ─────────────────────────────────────────

def compare_retrieval_strategies(
    max_samples: int = 100,
    top_k: int = 10,
) -> dict[str, RetrievalMetrics]:
    """
    A/B comparison: dense retrieval vs MMR retrieval.
    Useful for justifying the MMR choice in the thesis.
    """
    from src.agents.retriever import RetrieverAgent

    logger.info("Comparing Dense vs MMR retrieval strategies…")
    results = {}

    for strategy, use_mmr in [("dense", False), ("mmr", True)]:
        logger.info(f"  Evaluating: {strategy} (use_mmr={use_mmr})")
        ret = RetrieverAgent(top_k=top_k, use_mmr=use_mmr)
        metrics = evaluate_retriever(
            retriever=ret,
            top_k_values=[1, 3, 5, 10],
            max_samples=max_samples,
            log_to_mlflow=False,
        )
        results[strategy] = metrics

    if "dense" in results and "mmr" in results:
        d = results["dense"]
        m = results["mmr"]
        logger.info(
            f"\nDense vs MMR comparison:\n"
            f"  MRR:       {d.mrr:.4f} → {m.mrr:.4f} ({'↑' if m.mrr > d.mrr else '↓'})\n"
            f"  Recall@10: {d.recall_at_k.get(10, 0):.4f} → {m.recall_at_k.get(10, 0):.4f}\n"
            f"  NDCG@10:   {d.ndcg_at_k.get(10, 0):.4f} → {m.ndcg_at_k.get(10, 0):.4f}"
        )

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval evaluation")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--compare-strategies", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    settings.ensure_dirs()

    if args.compare_strategies:
        compare_retrieval_strategies(max_samples=args.max_samples, top_k=max(args.top_k))
    else:
        evaluate_retriever(
            top_k_values=args.top_k,
            max_samples=args.max_samples,
            log_to_mlflow=not args.no_mlflow,
        )


if __name__ == "__main__":
    main()
