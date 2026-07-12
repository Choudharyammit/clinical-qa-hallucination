"""
End-to-end pipeline test on 100 MedQA questions.

Runs the complete Clinical QA pipeline (retrieve → rerank → generate →
hallucination check → cite) on a sample of MedQA test questions and
reports detailed performance metrics.

This script validates:
  1. The full agent pipeline works end-to-end
  2. Response times are within target (<3s p95)
  3. Hallucination rates are tracked per sample
  4. Accuracy matches expected range

Usage:
    python scripts/run_pipeline.py --n-questions 100
    python scripts/run_pipeline.py --n-questions 100 --condition all
    python scripts/run_pipeline.py --n-questions 10 --fast   # Quick smoke test
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings


def run_e2e_pipeline_test(
    n_questions: int = 100,
    use_rag: bool = True,
    use_hallucination_filter: bool = True,
    condition_name: str = "rag_with_hallucination_filter",
    output_file: Path = None,
) -> dict:
    """
    Run end-to-end pipeline test on MedQA questions.
    Returns summary metrics dict.
    """
    from src.agents.orchestrator import OrchestratorAgent

    orchestrator = OrchestratorAgent()

    # Load test data
    test_path = settings.data_processed_dir / "medqa_test.jsonl"
    if not test_path.exists():
        test_path = settings.data_raw_dir / "medqa" / "test.jsonl"
    if not test_path.exists():
        logger.error("MedQA test data not found. Run: make download-data && make ingest-data")
        return {}

    samples = []
    with open(test_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    samples = samples[:n_questions]

    logger.info(
        f"\nE2E Pipeline Test\n"
        f"  Condition: {condition_name}\n"
        f"  Questions: {len(samples)}\n"
        f"  RAG: {use_rag} | Halluc filter: {use_hallucination_filter}\n"
    )

    correct = 0
    response_times = []
    hallucination_rates = []
    retrieval_counts = []
    iteration_counts = []
    results = []
    errors = 0

    start_wall = time.perf_counter()

    for i, sample in enumerate(tqdm(samples, desc=f"Pipeline [{condition_name}]")):
        question = sample.get("question", "")
        options = sample.get("options", {})
        gold = (sample.get("answer_idx") or sample.get("answer", "")).strip()
        if gold and len(gold) > 1:
            gold = gold[0].upper()

        try:
            result = orchestrator.run(
                query=question,
                mcq_options=options if isinstance(options, dict) else None,
                use_rag=use_rag,
                use_hallucination_filter=use_hallucination_filter,
            )

            prediction = (result.get("mcq_prediction") or "").upper()
            elapsed = result.get("response_time_s", 0)
            halluc = result.get("hallucination_rate", 0)
            sources = result.get("sources_retrieved", 0)
            iters = result.get("iterations", 1)

            is_correct = prediction == gold
            if is_correct:
                correct += 1

            response_times.append(elapsed)
            hallucination_rates.append(halluc)
            retrieval_counts.append(sources)
            iteration_counts.append(iters)

            results.append({
                "idx": i,
                "question": question[:120],
                "gold": gold,
                "prediction": prediction,
                "correct": is_correct,
                "response_time_s": round(elapsed, 3),
                "hallucination_rate": round(halluc, 4),
                "sources_retrieved": sources,
                "iterations": iters,
                "n_citations": len(result.get("citations", [])),
            })

        except Exception as e:
            logger.warning(f"  Sample {i} FAILED: {e}")
            errors += 1
            results.append({
                "idx": i, "question": question[:120], "gold": gold,
                "prediction": "ERROR", "correct": False, "error": str(e),
            })

    total_wall = time.perf_counter() - start_wall
    n_valid = len(samples) - errors

    # Compute metrics
    accuracy = correct / len(samples) if samples else 0.0
    avg_halluc = float(np.mean(hallucination_rates)) if hallucination_rates else 0.0
    avg_time = float(np.mean(response_times)) if response_times else 0.0
    p95_time = float(np.percentile(response_times, 95)) if response_times else 0.0
    p99_time = float(np.percentile(response_times, 99)) if response_times else 0.0
    avg_sources = float(np.mean(retrieval_counts)) if retrieval_counts else 0.0
    avg_iters = float(np.mean(iteration_counts)) if iteration_counts else 0.0

    # SLA check
    sla_met = p95_time < 3.0

    metrics = {
        "condition": condition_name,
        "n_questions": len(samples),
        "n_errors": errors,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "hallucination_rate_mean": round(avg_halluc, 4),
        "response_time_mean_s": round(avg_time, 3),
        "response_time_p95_s": round(p95_time, 3),
        "response_time_p99_s": round(p99_time, 3),
        "sources_retrieved_mean": round(avg_sources, 1),
        "react_iterations_mean": round(avg_iters, 2),
        "total_wall_time_s": round(total_wall, 1),
        "sla_3s_p95_met": sla_met,
    }

    # Print results table
    sla_icon = "✅" if sla_met else "❌"
    acc_icon = "✅" if accuracy >= 0.75 else ("🟡" if accuracy >= 0.5 else "❌")
    halluc_icon = "✅" if avg_halluc <= 0.15 else ("🟡" if avg_halluc <= 0.30 else "❌")

    logger.success(
        f"\n{'═'*60}\n"
        f"  E2E Pipeline Results — {condition_name}\n"
        f"{'─'*60}\n"
        f"  Questions tested:         {len(samples)}\n"
        f"  Errors:                   {errors}\n"
        f"  Accuracy:          {acc_icon}    {accuracy:.4f} ({correct}/{len(samples)})\n"
        f"  Hallucination Rate:{halluc_icon}    {avg_halluc:.4f}\n"
        f"  Avg response time:        {avg_time:.2f}s\n"
        f"  P95 response time: {sla_icon}    {p95_time:.2f}s  (SLA: <3s)\n"
        f"  P99 response time:        {p99_time:.2f}s\n"
        f"  Avg sources retrieved:    {avg_sources:.1f}\n"
        f"  Avg ReAct iterations:     {avg_iters:.2f}\n"
        f"  Total wall time:          {total_wall:.0f}s\n"
        f"{'═'*60}"
    )

    # Save results
    out = output_file or settings.data_processed_dir / f"e2e_{condition_name}_results.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary_out = settings.data_processed_dir / f"e2e_{condition_name}_metrics.json"
    with open(summary_out, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Results saved → {out}")
    return metrics


def run_all_conditions(n_questions: int = 100) -> dict[str, dict]:
    """Run all 3 experimental conditions and print comparison."""
    all_results = {}

    conditions = [
        ("base_llm", False, False),
        ("rag_only", True, False),
        ("rag_with_hallucination_filter", True, True),
    ]

    for condition, use_rag, use_halluc in conditions:
        logger.info(f"\n{'#'*60}\nStarting condition: {condition}\n{'#'*60}")
        all_results[condition] = run_e2e_pipeline_test(
            n_questions=n_questions,
            use_rag=use_rag,
            use_hallucination_filter=use_halluc,
            condition_name=condition,
        )

    # Print comparison
    print_comparison_table(all_results)
    return all_results


def print_comparison_table(results: dict[str, dict]) -> None:
    if not results:
        return
    hdr = f"\n{'═'*75}\n{'Condition':<35} {'Accuracy':>10} {'Halluc':>10} {'P95 Time':>10}\n{'─'*75}"
    rows = []
    for name, m in results.items():
        rows.append(
            f"{name:<35} {m.get('accuracy', 0):>10.4f} "
            f"{m.get('hallucination_rate_mean', 0):>10.4f} "
            f"{m.get('response_time_p95_s', 0):>9.2f}s"
        )
    logger.info(hdr + "\n" + "\n".join(rows) + f"\n{'═'*75}")


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end pipeline test")
    parser.add_argument("--n-questions", type=int, default=100)
    parser.add_argument("--condition", default="rag_with_hallucination_filter",
                        choices=["base_llm", "rag_only", "rag_with_hallucination_filter", "all"])
    parser.add_argument("--fast", action="store_true",
                        help="Quick smoke test: 10 questions only")
    args = parser.parse_args()

    settings.ensure_dirs()

    n = 10 if args.fast else args.n_questions

    if args.condition == "all":
        run_all_conditions(n)
    elif args.condition == "base_llm":
        run_e2e_pipeline_test(n, use_rag=False, use_hallucination_filter=False,
                              condition_name="base_llm")
    elif args.condition == "rag_only":
        run_e2e_pipeline_test(n, use_rag=True, use_hallucination_filter=False,
                              condition_name="rag_only")
    else:
        run_e2e_pipeline_test(n, use_rag=True, use_hallucination_filter=True,
                              condition_name="rag_with_hallucination_filter")


if __name__ == "__main__":
    main()
