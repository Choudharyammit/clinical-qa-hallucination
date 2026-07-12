"""
Hallucination rate evaluator.

Measures hallucination rates across conditions by running the
HallucinationDetectorAgent on generated responses and computing
aggregate statistics. Produces comparison charts.

Usage:
    python -m src.evaluation.hallucination_eval
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger

from src.config import settings


def load_predictions(condition: str) -> list[dict]:
    """Load prediction JSONL for a given condition."""
    path = settings.data_processed_dir / f"medqa_predictions_{condition}.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def compute_hallucination_stats(predictions: list[dict]) -> dict:
    """Compute hallucination statistics from prediction records."""
    rates = [p.get("hallucination_rate", 0) for p in predictions]
    if not rates:
        return {}
    return {
        "mean": float(np.mean(rates)),
        "median": float(np.median(rates)),
        "std": float(np.std(rates)),
        "p25": float(np.percentile(rates, 25)),
        "p75": float(np.percentile(rates, 75)),
        "p95": float(np.percentile(rates, 95)),
        "n": len(rates),
        "pct_zero_halluc": float(np.mean([r == 0 for r in rates])),
    }


def plot_hallucination_comparison(
    results: dict[str, dict],
    output_path: Optional[Path] = None,
) -> None:
    """
    Generate comparison bar chart of hallucination rates across conditions.
    Saves to docs/ directory.
    """
    conditions = list(results.keys())
    means = [results[c]["mean"] for c in conditions]
    stds = [results[c]["std"] for c in conditions]

    label_map = {
        "base_llm": "Base LLM\n(No RAG)",
        "rag_only": "RAG Only\n(No Filter)",
        "rag_with_hallucination_filter": "RAG +\nHalluc Filter\n(Proposed)",
    }
    labels = [label_map.get(c, c) for c in conditions]
    colors = ["#d73027", "#fee090", "#1a9850"][:len(conditions)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    ax = axes[0]
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    ax.set_ylabel("Hallucination Rate (fraction of claims ungrounded)", fontsize=11)
    ax.set_title("Hallucination Rate Comparison Across Conditions", fontsize=13, fontweight="bold")
    ax.set_ylim(0, min(1.0, max(means) * 1.5))
    ax.grid(axis="y", alpha=0.3)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{mean:.3f}", ha="center", va="bottom", fontweight="bold")

    # Distribution violin / box plot
    ax2 = axes[1]
    all_data = []
    all_labels = []
    for c in conditions:
        preds = load_predictions(c)
        rates = [p.get("hallucination_rate", 0) for p in preds]
        if rates:
            all_data.append(rates)
            all_labels.append(label_map.get(c, c))

    if all_data:
        ax2.boxplot(all_data, labels=all_labels, patch_artist=True,
                    boxprops=dict(facecolor="lightblue", alpha=0.7))
        ax2.set_ylabel("Hallucination Rate", fontsize=11)
        ax2.set_title("Hallucination Rate Distribution", fontsize=13, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = output_path or Path("docs") / "hallucination_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Hallucination comparison chart saved → {out}")


def run_hallucination_evaluation(output_dir: Optional[Path] = None) -> dict:
    """Main evaluation function."""
    conditions = ["base_llm", "rag_only", "rag_with_hallucination_filter"]
    results = {}

    for condition in conditions:
        preds = load_predictions(condition)
        if not preds:
            logger.warning(f"No predictions for condition: {condition}. Run medqa_eval first.")
            continue
        stats = compute_hallucination_stats(preds)
        results[condition] = stats
        logger.info(
            f"[{condition}] Mean hallucination rate: {stats.get('mean', 0):.4f} "
            f"± {stats.get('std', 0):.4f} (n={stats.get('n', 0)})"
        )

    if len(results) >= 2:
        conditions_present = list(results.keys())
        if "base_llm" in results and "rag_with_hallucination_filter" in results:
            base = results["base_llm"]["mean"]
            proposed = results["rag_with_hallucination_filter"]["mean"]
            reduction_pct = (base - proposed) / base * 100 if base > 0 else 0
            logger.success(
                f"\nHallucination Reduction (Proposed vs Base): "
                f"{reduction_pct:.1f}% "
                f"({'✅ MEETS' if reduction_pct >= 50 else '❌ BELOW'} 50% target)"
            )
        plot_hallucination_comparison(results, output_dir)

    return results


def main():
    parser = argparse.ArgumentParser(description="Hallucination rate evaluation")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    settings.ensure_dirs()
    run_hallucination_evaluation(args.output_dir)


if __name__ == "__main__":
    main()
