"""
Hallucination evaluation script — Day 4.

Modes:
  --mode stats      Load pre-computed predictions and compute stats (fast, no inference)
  --mode ab-test    Run live A/B test (requires models loaded)
  --mode both       Stats from pre-computed + run A/B if no saved results

All results logged to MLflow.

Usage:
    python scripts/run_hallucination_eval.py --mode stats
    python scripts/run_hallucination_eval.py --mode ab-test --n 50
    python scripts/run_hallucination_eval.py --mode both --n 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
from loguru import logger

from src.config import settings
from src.evaluation.hallucination_eval import (
    compute_hallucination_stats,
    load_predictions,
    plot_hallucination_comparison,
    run_hallucination_evaluation,
)
from src.evaluation.ab_test import (
    ABTestResults,
    load_ab_test_results,
    run_ab_test,
)
from src.utils.logging import setup_logging


CONDITIONS = ["base_llm", "rag_only", "rag_with_hallucination_filter"]


def run_stats_mode() -> dict:
    """Load pre-computed predictions and log hallucination stats to MLflow."""
    logger.info("Mode: stats — loading pre-computed predictions")
    results = run_hallucination_evaluation(output_dir=Path("docs"))

    if not results:
        logger.warning("No prediction files found. Run medqa_eval first:")
        logger.warning("  python scripts/run_pipeline.py --condition all --n-questions 200")
        return {}

    # Log each condition separately to MLflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    for condition, stats in results.items():
        with mlflow.start_run(run_name=f"halluc_stats_{condition}"):
            mlflow.log_params({"condition": condition, "eval_mode": "stats"})
            mlflow.log_metrics({
                "hallucination_rate": stats.get("mean", 0.0),
                "hallucination_std": stats.get("std", 0.0),
                "hallucination_p95": stats.get("p95", 0.0),
                "pct_zero_halluc": stats.get("pct_zero_halluc", 0.0),
                "n_samples": float(stats.get("n", 0)),
            })
        logger.info(f"Logged {condition} stats to MLflow")

    # Compute and log hallucination reduction
    if "base_llm" in results and "rag_with_hallucination_filter" in results:
        base_h = results["base_llm"]["mean"]
        prop_h = results["rag_with_hallucination_filter"]["mean"]
        reduction = (base_h - prop_h) / base_h * 100 if base_h > 0 else 0.0

        with mlflow.start_run(run_name="hallucination_reduction_summary"):
            mlflow.log_metrics({
                "base_llm_hallucination_rate": base_h,
                "proposed_hallucination_rate": prop_h,
                "hallucination_reduction_pct": reduction,
                "meets_50pct_target": float(reduction >= 50.0),
            })

        print(f"\n  Hallucination reduction: {reduction:.1f}%  "
              f"{'✅ MEETS' if reduction >= 50.0 else '❌ BELOW'} 50% target")

    return results


def run_ab_mode(n: int = 100, load_questions_n: int = 500) -> ABTestResults:
    """Run live A/B test using the orchestrator."""
    logger.info(f"Mode: ab-test — {n} questions × 2 conditions")

    # Check for saved results first
    saved = load_ab_test_results()
    if saved and saved.n >= n:
        logger.info(f"Found saved A/B results (n={saved.n}). Loading instead of re-running.")
        saved.print_summary()
        return saved

    # Load MedQA questions
    test_path = settings.data_raw_dir / "medqa" / "test.jsonl"
    if not test_path.exists():
        logger.error(f"MedQA test file not found: {test_path}")
        logger.error("Run: python src/data/downloader.py --dataset medqa")
        sys.exit(1)

    questions: list[dict] = []
    with open(test_path) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
            if len(questions) >= load_questions_n:
                break

    logger.info(f"Loaded {len(questions)} MedQA questions")

    # Lazy-load orchestrator
    from src.agents.orchestrator import OrchestratorAgent
    logger.info("Initializing orchestrator (this may take a minute)…")
    orchestrator = OrchestratorAgent()

    results = run_ab_test(
        questions=questions,
        orchestrator=orchestrator,
        n=n,
        log_mlflow=True,
    )
    results.print_summary()
    return results


def print_quick_stats() -> None:
    """Print hallucination rate summary from any available data."""
    print(f"\n{'─'*60}")
    print("  Hallucination Rate Summary (from pre-computed predictions)")
    print(f"{'─'*60}")

    label_map = {
        "base_llm": "Base LLM (No RAG)",
        "rag_only": "RAG Only (No Filter)",
        "rag_with_hallucination_filter": "RAG + Halluc Filter (Proposed)",
    }

    found_any = False
    for cond in CONDITIONS:
        preds = load_predictions(cond)
        if preds:
            stats = compute_hallucination_stats(preds)
            print(f"  {label_map.get(cond, cond)}")
            print(f"    n={stats['n']}, mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                  f"p95={stats['p95']:.4f}, pct_zero={stats['pct_zero_halluc']:.2%}")
            found_any = True

    if not found_any:
        print("  No prediction files found. Run: python scripts/run_pipeline.py --condition all")

    # A/B test summary
    saved = load_ab_test_results()
    if saved and saved.n > 0:
        print(f"\n  A/B Test Results (n={saved.n}):")
        print(f"    Hallucination reduction: {saved.hallucination_reduction_pct:.1f}%  "
              f"{'✅ MEETS' if saved.meets_50pct_target() else '❌ BELOW'} 50% target")
        print(f"    Welch t={saved.t_statistic:.3f}, p={saved.p_value:.4f}  "
              f"({'significant' if saved.significant else 'not significant'})")

    print(f"{'─'*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hallucination evaluation — stats and/or live A/B test"
    )
    parser.add_argument(
        "--mode", choices=["stats", "ab-test", "both"], default="stats",
        help="Evaluation mode (default: stats)"
    )
    parser.add_argument("--n", type=int, default=100, help="Questions for A/B test")
    parser.add_argument("--summary", action="store_true", help="Print quick summary and exit")
    args = parser.parse_args()

    setup_logging()
    settings.ensure_dirs()

    if args.summary:
        print_quick_stats()
        return

    if args.mode == "stats":
        run_stats_mode()
    elif args.mode == "ab-test":
        run_ab_mode(n=args.n)
    else:  # both
        run_stats_mode()
        run_ab_mode(n=args.n)


if __name__ == "__main__":
    main()
