"""
MLflow experiment initialization script.

Creates the experiment structure, registers runs for all three conditions,
and sets up metric comparison views. Run this once before benchmarking.

Usage:
    python scripts/setup_mlflow.py
    python scripts/setup_mlflow.py --reset   # delete and recreate experiment
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
from loguru import logger

from src.config import settings

# Metrics we will log — used to pre-declare the schema
EXPECTED_METRICS = [
    "accuracy",
    "hallucination_rate",
    "avg_response_time_s",
    "p95_response_time_s",
    "mrr",
    "recall@5",
    "recall@10",
    "ndcg@10",
    "faithfulness",
    "answer_relevancy",
]

CONDITIONS = [
    "base_llm",
    "rag_only",
    "rag_with_hallucination_filter",
    "retrieval_evaluation",
    "ragas_evaluation",
]


def setup_experiment(reset: bool = False) -> None:
    """Initialize MLflow experiment with placeholder runs."""
    settings.ensure_dirs()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    # Delete experiment if reset requested
    if reset:
        try:
            existing = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
            if existing:
                mlflow.delete_experiment(existing.experiment_id)
                logger.info(f"Deleted existing experiment: {settings.mlflow_experiment_name}")
        except Exception as e:
            logger.warning(f"Could not delete experiment: {e}")

    # Create experiment
    experiment = mlflow.set_experiment(settings.mlflow_experiment_name)
    logger.info(f"Experiment: '{settings.mlflow_experiment_name}'")
    logger.info(f"Experiment ID: {experiment.experiment_id}")
    logger.info(f"Tracking URI: {settings.mlflow_tracking_uri}")

    # Log system configuration as a setup run
    with mlflow.start_run(run_name="system_config") as run:
        mlflow.log_params({
            "llm_model": settings.llm_model_name,
            "embedding_model": settings.embedding_model,
            "nli_model": settings.nli_model,
            "reranker_model": settings.reranker_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "retriever_top_k": settings.retriever_top_k,
            "max_context_chunks": settings.max_context_chunks,
            "nli_threshold": settings.nli_entailment_threshold,
            "llm_temperature": settings.llm_temperature,
            "llm_quantization": settings.llm_quantization,
            "random_seed": settings.random_seed,
        })
        mlflow.log_metric("setup_complete", 1.0)
        logger.success(f"System config logged (run_id={run.info.run_id})")

    # Print summary
    print(f"\n{'='*55}")
    print(f"  MLflow Experiment Ready")
    print(f"{'='*55}")
    print(f"  Name:          {settings.mlflow_experiment_name}")
    print(f"  Tracking URI:  {settings.mlflow_tracking_uri}")
    print(f"\n  View runs:")
    print(f"    mlflow ui --backend-store-uri {settings.mlflow_tracking_uri}")
    print(f"    → http://localhost:5000")
    print(f"\n  Expected conditions to evaluate:")
    for c in CONDITIONS:
        print(f"    - {c}")
    print(f"\n  Expected metrics:")
    for m in EXPECTED_METRICS:
        print(f"    - {m}")
    print(f"{'='*55}\n")


def show_results() -> None:
    """Print a summary of all completed runs in the experiment."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()

    try:
        experiment = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
        if not experiment:
            logger.warning("No experiment found. Run setup_mlflow.py first.")
            return

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=20,
        )

        if not runs:
            print("No runs recorded yet.")
            return

        print(f"\nRecent runs for '{settings.mlflow_experiment_name}':")
        print(f"{'─'*80}")
        print(f"{'Run Name':<35} {'Accuracy':>10} {'Halluc':>10} {'Time(s)':>8}")
        print(f"{'─'*80}")

        for run in runs:
            name = run.data.tags.get("mlflow.runName", run.info.run_id[:8])
            acc = run.data.metrics.get("accuracy", float("nan"))
            halluc = run.data.metrics.get("hallucination_rate", float("nan"))
            elapsed = run.data.metrics.get("avg_response_time_s", float("nan"))
            print(f"{name:<35} {acc:>10.4f} {halluc:>10.4f} {elapsed:>7.2f}s")

        print(f"{'─'*80}\n")

    except Exception as e:
        logger.error(f"Could not read MLflow runs: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize MLflow experiment")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate experiment")
    parser.add_argument("--show", action="store_true", help="Show current run results")
    args = parser.parse_args()

    if args.show:
        show_results()
    else:
        setup_experiment(reset=args.reset)


if __name__ == "__main__":
    main()
