"""
Full benchmark runner — Day 5.

Orchestrates all evaluation phases in sequence:
  1. MedQA accuracy: 500 questions × 3 conditions (base_llm, rag_only, rag_with_hallucination_filter)
  2. Hallucination stats: aggregate from predictions
  3. RAGAS: faithfulness, answer_relevancy, context_precision, context_recall
  4. Retrieval metrics: Recall@K, MRR, NDCG@K
  5. Comparison table: printed to stdout and saved as CSV
  6. Charts: saved to docs/

All metrics are logged to MLflow. Results are saved to data/processed/.

Usage:
    python scripts/run_full_benchmark.py                    # full 500q benchmark
    python scripts/run_full_benchmark.py --n 50             # quick 50q run
    python scripts/run_full_benchmark.py --skip-inference   # charts only from saved results
    python scripts/run_full_benchmark.py --phase medqa      # single phase
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import numpy as np
from loguru import logger

from src.config import settings
from src.utils.logging import setup_logging


CONDITIONS = [
    ("base_llm",                     False, False),
    ("rag_only",                     True,  False),
    ("rag_with_hallucination_filter", True,  True),
]

CONDITION_LABELS = {
    "base_llm":                     "Base LLM (No RAG)",
    "rag_only":                     "RAG Only (No Filter)",
    "rag_with_hallucination_filter": "RAG + Hallucination Filter (Proposed)",
}


# ── Phase runners ──────────────────────────────────────────────────────────────

def phase_medqa(n: int = 500) -> dict[str, dict]:
    """Run MedQA evaluation across all 3 conditions. Returns {condition: metrics}."""
    from src.evaluation.medqa_eval import run_medqa_evaluation

    results: dict[str, dict] = {}
    for condition, use_rag, use_halluc in CONDITIONS:
        logger.info(f"\n{'─'*60}")
        logger.info(f"MedQA Phase: condition={condition} (n={n})")
        logger.info(f"{'─'*60}")
        try:
            metrics = run_medqa_evaluation(
                condition=condition,
                max_samples=n,
                use_rag=use_rag,
                use_hallucination_filter=use_halluc,
            )
            results[condition] = metrics
        except Exception as e:
            logger.error(f"MedQA eval failed for {condition}: {e}")
            results[condition] = {}
    return results


def phase_hallucination() -> dict[str, dict]:
    """Compute hallucination stats from saved prediction files."""
    from src.evaluation.hallucination_eval import (
        compute_hallucination_stats,
        load_predictions,
        plot_hallucination_comparison,
    )

    results: dict[str, dict] = {}
    for condition, _, _ in CONDITIONS:
        preds = load_predictions(condition)
        if preds:
            stats = compute_hallucination_stats(preds)
            results[condition] = stats
        else:
            logger.warning(f"No predictions for {condition} — skipping hallucination stats")

    if len(results) >= 2:
        plot_hallucination_comparison(results, output_path=Path("docs") / "hallucination_comparison.png")

    return results


def phase_ragas(n: int = 100) -> dict:
    """Run RAGAS evaluation on proposed system."""
    from src.evaluation.ragas_eval import run_ragas_evaluation
    try:
        return run_ragas_evaluation(max_samples=n, condition="rag_with_hallucination_filter")
    except Exception as e:
        logger.warning(f"RAGAS phase failed (non-fatal): {e}")
        return {}


def phase_retrieval(n: int = 200) -> dict:
    """Run retrieval metric evaluation."""
    from src.evaluation.retrieval_eval import evaluate_retriever, compare_retrieval_strategies
    try:
        results = compare_retrieval_strategies(max_samples=n)
        return {name: vars(m) for name, m in results.items()}
    except Exception as e:
        logger.warning(f"Retrieval eval phase failed (non-fatal): {e}")
        return {}


# ── Reporting ──────────────────────────────────────────────────────────────────

def build_comparison_table(
    medqa_results: dict[str, dict],
    halluc_results: dict[str, dict],
    ragas_metrics: dict,
) -> list[dict]:
    """Build a unified comparison row per condition."""
    rows = []
    for condition, _, _ in CONDITIONS:
        mq = medqa_results.get(condition, {})
        h = halluc_results.get(condition, {})
        row = {
            "Condition": CONDITION_LABELS.get(condition, condition),
            "MedQA Accuracy": mq.get("accuracy", None),
            "Hallucination Rate": h.get("mean", mq.get("hallucination_rate", None)),
            "Halluc Std": h.get("std", None),
            "Avg Time (s)": mq.get("avg_response_time_s", None),
            "P95 Time (s)": mq.get("p95_response_time_s", None),
            "N": mq.get("total_questions", None),
        }
        if condition == "rag_with_hallucination_filter" and ragas_metrics:
            row["RAGAS Faithfulness"] = ragas_metrics.get("faithfulness", None)
            row["RAGAS Relevancy"] = ragas_metrics.get("answer_relevancy", None)
        rows.append(row)
    return rows


def print_comparison_table(rows: list[dict]) -> None:
    """Print a formatted comparison table to stdout."""
    print(f"\n{'═'*90}")
    print("  Full Benchmark Results — Thesis Table 4.1")
    print(f"{'═'*90}")

    headers = ["Condition", "Accuracy", "Halluc Rate", "Halluc Std", "Avg Time", "P95 Time", "N"]
    widths = [42, 10, 12, 12, 10, 10, 6]
    header_line = "  " + "".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    print(header_line)
    print("  " + "─" * (sum(widths)))

    for row in rows:
        def fmt(v, w):
            if v is None:
                return f"{'—':<{w}}"
            if isinstance(v, float):
                return f"{v:<{w}.4f}"
            return f"{str(v):<{w}}"

        line = "  " + "".join([
            fmt(row["Condition"], 42),
            fmt(row.get("MedQA Accuracy"), 10),
            fmt(row.get("Hallucination Rate"), 12),
            fmt(row.get("Halluc Std"), 12),
            fmt(row.get("Avg Time (s)"), 10),
            fmt(row.get("P95 Time (s)"), 10),
            fmt(row.get("N"), 6),
        ])
        print(line)

    print(f"{'═'*90}\n")

    # Hallucination reduction
    base = next((r for r in rows if "Base LLM" in r["Condition"]), None)
    prop = next((r for r in rows if "Proposed" in r["Condition"]), None)
    if base and prop and base.get("Hallucination Rate") and prop.get("Hallucination Rate"):
        bh = base["Hallucination Rate"]
        ph = prop["Hallucination Rate"]
        reduction = (bh - ph) / bh * 100 if bh > 0 else 0
        print(f"  Hallucination reduction (Base → Proposed): {reduction:.1f}%  "
              f"{'✅ MEETS' if reduction >= 50 else '❌ BELOW'} 50% target\n")


def save_comparison_table(rows: list[dict], output_path: Path) -> None:
    """Save comparison table to CSV."""
    import csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Comparison table saved → {output_path}")


def generate_benchmark_charts(
    medqa_results: dict[str, dict],
    halluc_results: dict[str, dict],
    ragas_metrics: dict,
    output_dir: Path,
) -> None:
    """Generate all thesis charts and save to output_dir."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    available_conditions = [c for c, _, _ in CONDITIONS if medqa_results.get(c)]
    if not available_conditions:
        logger.warning("No results to chart yet.")
        return

    labels = [CONDITION_LABELS[c].replace(" (Proposed)", "\n(Proposed)") for c in available_conditions]
    colors = ["#d73027", "#fee090", "#1a9850"][:len(available_conditions)]

    accs = [medqa_results[c].get("accuracy", 0) for c in available_conditions]
    halluc_means = [
        halluc_results.get(c, {}).get("mean", medqa_results[c].get("hallucination_rate", 0))
        for c in available_conditions
    ]
    halluc_stds = [halluc_results.get(c, {}).get("std", 0) for c in available_conditions]
    avg_times = [medqa_results[c].get("avg_response_time_s", 0) for c in available_conditions]
    p95_times = [medqa_results[c].get("p95_response_time_s", 0) for c in available_conditions]

    # ── Figure 1: 4-panel main results ──────────────────────────────────────
    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    # (a) Accuracy
    ax0 = fig.add_subplot(gs[0, 0])
    bars = ax0.bar(range(len(labels)), accs, color=colors, alpha=0.85, edgecolor="black")
    ax0.axhline(0.75, color="red", linestyle="--", lw=1.5, label="Target 75%")
    ax0.set_xticks(range(len(labels)))
    ax0.set_xticklabels(labels, fontsize=9)
    ax0.set_ylabel("MedQA Accuracy")
    ax0.set_title("(a) MedQA Accuracy by Condition", fontweight="bold")
    ax0.set_ylim(0, 1.0)
    ax0.legend(fontsize=9)
    for i, a in enumerate(accs):
        ax0.text(i, a + 0.01, f"{a:.3f}", ha="center", fontsize=9, fontweight="bold")

    # (b) Hallucination rate
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(range(len(labels)), halluc_means, yerr=halluc_stds,
            capsize=5, color=colors, alpha=0.85, edgecolor="black")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Hallucination Rate (fraction)")
    ax1.set_title("(b) Hallucination Rate by Condition", fontweight="bold")
    for i, h in enumerate(halluc_means):
        ax1.text(i, h + max(halluc_stds) * 0.1 + 0.005, f"{h:.3f}",
                 ha="center", fontsize=9, fontweight="bold")

    # (c) Response time
    ax2 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(labels))
    w = 0.35
    ax2.bar(x - w / 2, avg_times, w, label="Mean", color="steelblue", alpha=0.8)
    ax2.bar(x + w / 2, p95_times, w, label="P95", color="coral", alpha=0.8)
    ax2.axhline(3.0, color="red", linestyle="--", lw=1.5, label="SLA 3s")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Time (s)")
    ax2.set_title("(c) Response Time (Mean & P95)", fontweight="bold")
    ax2.legend(fontsize=9)

    # (d) RAGAS metrics (proposed system only)
    ax3 = fig.add_subplot(gs[1, 1])
    if ragas_metrics:
        ragas_labels = ["Faithfulness", "Answer\nRelevancy", "Context\nPrecision", "Context\nRecall"]
        ragas_vals = [
            ragas_metrics.get("faithfulness", 0),
            ragas_metrics.get("answer_relevancy", 0),
            ragas_metrics.get("context_precision", 0),
            ragas_metrics.get("context_recall", 0),
        ]
        ax3.bar(ragas_labels, ragas_vals, color="#1a9850", alpha=0.85, edgecolor="black")
        ax3.set_ylim(0, 1.0)
        ax3.set_ylabel("Score")
        ax3.set_title("(d) RAGAS Metrics (Proposed System)", fontweight="bold")
        for i, v in enumerate(ragas_vals):
            ax3.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    else:
        ax3.text(0.5, 0.5, "RAGAS metrics\nnot yet available",
                 ha="center", va="center", transform=ax3.transAxes, fontsize=12, color="gray")
        ax3.set_title("(d) RAGAS Metrics (Proposed System)", fontweight="bold")

    plt.suptitle("Clinical QA System — Full Benchmark Results\n(LJMU MSc ML&AI — Amit Choudhary 1187247)",
                 fontsize=13, fontweight="bold", y=1.01)
    chart_path = output_dir / "benchmark_results_figure.png"
    plt.savefig(chart_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.success(f"Main results chart saved → {chart_path}")

    # ── Figure 2: Hallucination distribution per condition ───────────────────
    from src.evaluation.hallucination_eval import load_predictions
    all_rates = {}
    for c in available_conditions:
        preds = load_predictions(c)
        if preds:
            all_rates[c] = [p.get("hallucination_rate", 0) for p in preds]

    if all_rates:
        fig2, ax = plt.subplots(figsize=(9, 5))
        for i, c in enumerate(available_conditions):
            rates = all_rates.get(c, [])
            if rates:
                ax.hist(rates, bins=25, alpha=0.6,
                        label=CONDITION_LABELS[c], color=colors[i], density=True)
        ax.set_xlabel("Hallucination Rate", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("Hallucination Rate Distribution by Condition", fontweight="bold", fontsize=13)
        ax.legend(fontsize=10)
        dist_path = output_dir / "hallucination_distribution.png"
        plt.savefig(dist_path, bbox_inches="tight", dpi=150)
        plt.close()
        logger.success(f"Hallucination distribution chart saved → {dist_path}")


def log_summary_to_mlflow(
    medqa_results: dict[str, dict],
    halluc_results: dict[str, dict],
    ragas_metrics: dict,
) -> None:
    """Log all final metrics to MLflow as a single summary run."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run(run_name="full_benchmark_summary"):
        for condition, _, _ in CONDITIONS:
            mq = medqa_results.get(condition, {})
            h = halluc_results.get(condition, {})
            prefix = condition[:6]  # shorten for metric names
            mlflow.log_metrics({k: v for k, v in {
                f"{prefix}_accuracy": mq.get("accuracy"),
                f"{prefix}_hallucination_rate": h.get("mean", mq.get("hallucination_rate")),
                f"{prefix}_avg_time_s": mq.get("avg_response_time_s"),
                f"{prefix}_p95_time_s": mq.get("p95_response_time_s"),
            }.items() if v is not None})

        if ragas_metrics:
            mlflow.log_metrics({k: v for k, v in ragas_metrics.items()
                                 if isinstance(v, float)})

        # Compute hallucination reduction
        base_h = halluc_results.get("base_llm", {}).get("mean") or \
                 medqa_results.get("base_llm", {}).get("hallucination_rate", 0)
        prop_h = halluc_results.get("rag_with_hallucination_filter", {}).get("mean") or \
                 medqa_results.get("rag_with_hallucination_filter", {}).get("hallucination_rate", 0)
        if base_h and prop_h:
            reduction = (base_h - prop_h) / base_h * 100 if base_h > 0 else 0
            mlflow.log_metric("hallucination_reduction_pct", reduction)
            mlflow.log_metric("meets_50pct_target", float(reduction >= 50.0))

        mlflow.log_param("benchmark_timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))

    logger.success("Full benchmark summary logged to MLflow.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Full benchmark runner — Day 5")
    parser.add_argument("--n", type=int, default=500, help="MedQA questions per condition")
    parser.add_argument("--ragas-n", type=int, default=100, help="PubMedQA samples for RAGAS")
    parser.add_argument("--retrieval-n", type=int, default=200, help="Samples for retrieval eval")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip inference — just recompute stats from saved predictions")
    parser.add_argument("--phase", choices=["medqa", "hallucination", "ragas", "retrieval", "charts", "all"],
                        default="all", help="Run a single phase only")
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()

    setup_logging()
    settings.ensure_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    medqa_results: dict[str, dict] = {}
    halluc_results: dict[str, dict] = {}
    ragas_metrics: dict = {}
    retrieval_results: dict = {}

    total_start = time.time()

    # ── MedQA phase ────────────────────────────────────────────────────────
    if args.phase in ("medqa", "all") and not args.skip_inference:
        logger.info(f"=== Phase 1: MedQA Benchmark ({args.n} questions × 3 conditions) ===")
        medqa_results = phase_medqa(n=args.n)
    else:
        # Load from saved files
        from src.evaluation.hallucination_eval import load_predictions
        for condition, _, _ in CONDITIONS:
            preds = load_predictions(condition)
            if preds:
                from src.evaluation.hallucination_eval import compute_hallucination_stats
                stats = compute_hallucination_stats(preds)
                accs = [p.get("correct", False) for p in preds]
                times = [p.get("response_time_s", 0) for p in preds]
                medqa_results[condition] = {
                    "accuracy": float(np.mean(accs)) if accs else 0.0,
                    "hallucination_rate": stats.get("mean", 0.0),
                    "avg_response_time_s": float(np.mean(times)) if times else 0.0,
                    "p95_response_time_s": float(np.percentile(times, 95)) if times else 0.0,
                    "total_questions": len(preds),
                }

    # ── Hallucination phase ────────────────────────────────────────────────
    if args.phase in ("hallucination", "all"):
        logger.info("=== Phase 2: Hallucination Statistics ===")
        halluc_results = phase_hallucination()

    # ── RAGAS phase ────────────────────────────────────────────────────────
    if args.phase in ("ragas", "all") and not args.skip_inference:
        logger.info(f"=== Phase 3: RAGAS Evaluation ({args.ragas_n} samples) ===")
        ragas_metrics = phase_ragas(n=args.ragas_n)

    # ── Retrieval phase ────────────────────────────────────────────────────
    if args.phase in ("retrieval", "all") and not args.skip_inference:
        logger.info(f"=== Phase 4: Retrieval Metrics ({args.retrieval_n} samples) ===")
        retrieval_results = phase_retrieval(n=args.retrieval_n)

    # ── Charts and tables ──────────────────────────────────────────────────
    if args.phase in ("charts", "all"):
        logger.info("=== Phase 5: Charts and Comparison Table ===")
        rows = build_comparison_table(medqa_results, halluc_results, ragas_metrics)
        print_comparison_table(rows)
        save_comparison_table(rows, args.output_dir / "comparison_table.csv")
        generate_benchmark_charts(medqa_results, halluc_results, ragas_metrics, args.output_dir)
        log_summary_to_mlflow(medqa_results, halluc_results, ragas_metrics)

    elapsed = time.time() - total_start
    logger.success(f"\n✅ Full benchmark complete in {elapsed/60:.1f} minutes.")
    logger.success(f"   Charts and tables saved to: {args.output_dir}/")
    logger.success(f"   View MLflow: mlflow ui --backend-store-uri {settings.mlflow_tracking_uri}")


if __name__ == "__main__":
    main()
