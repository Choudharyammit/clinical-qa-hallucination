"""
MedQA accuracy evaluator.

Runs the full Clinical QA pipeline on the MedQA test set (USMLE-style MCQs)
and computes per-condition accuracy. Logs all results to MLflow.

Usage:
    python -m src.evaluation.medqa_eval --max-samples 500
    python -m src.evaluation.medqa_eval --condition base_llm --max-samples 100
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
from loguru import logger
from tqdm import tqdm

from src.config import settings


def run_medqa_evaluation(
    condition: str = "rag_with_hallucination_filter",
    max_samples: int = 500,
    use_rag: bool = True,
    use_hallucination_filter: bool = True,
    output_file: Optional[Path] = None,
) -> dict:
    """
    Run MedQA accuracy evaluation for a single experimental condition.

    Returns dict with accuracy, hallucination_rate, avg_response_time, etc.
    """
    from src.agents.orchestrator import OrchestratorAgent

    orchestrator = OrchestratorAgent()

    # Load test data
    test_path = settings.data_processed_dir / "medqa_test.jsonl"
    if not test_path.exists():
        test_path = settings.data_raw_dir / "medqa" / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(
            f"MedQA test file not found. Run: make download-data first."
        )

    samples = []
    with open(test_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    samples = samples[:max_samples]
    logger.info(f"Evaluating {len(samples)} MedQA samples | condition={condition}")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    correct = 0
    total = 0
    response_times = []
    hallucination_rates = []
    predictions = []

    with mlflow.start_run(run_name=f"medqa_{condition}_{int(time.time())}"):
        mlflow.log_params({
            "condition": condition,
            "use_rag": use_rag,
            "use_hallucination_filter": use_hallucination_filter,
            "n_samples": len(samples),
            "model": settings.llm_model_name,
            "embedding_model": settings.embedding_model,
        })

        for i, sample in enumerate(tqdm(samples, desc=f"MedQA [{condition}]")):
            question = sample.get("question", "")
            options = sample.get("options", {})
            gold_answer = (sample.get("answer_idx") or sample.get("answer", "")).strip()
            if gold_answer and len(gold_answer) > 1:
                gold_answer = gold_answer[0].upper()

            try:
                result = orchestrator.run(
                    query=question,
                    mcq_options=options if isinstance(options, dict) else None,
                    use_rag=use_rag,
                    use_hallucination_filter=use_hallucination_filter,
                )
                prediction = (result.get("mcq_prediction") or "").upper()
                is_correct = prediction == gold_answer

                response_times.append(result.get("response_time_s", 0))
                hallucination_rates.append(result.get("hallucination_rate", 0))
                predictions.append({
                    "question": question[:100],
                    "gold": gold_answer,
                    "prediction": prediction,
                    "correct": is_correct,
                    "response_time_s": result.get("response_time_s", 0),
                    "hallucination_rate": result.get("hallucination_rate", 0),
                })

                if is_correct:
                    correct += 1
                total += 1

                # Log interim metrics
                if (i + 1) % 50 == 0:
                    interim_acc = correct / total
                    mlflow.log_metric("interim_accuracy", interim_acc, step=total)
                    logger.info(f"  Interim accuracy: {interim_acc:.3f} ({correct}/{total})")

            except Exception as e:
                logger.warning(f"Sample {i} failed: {e}")
                total += 1
                predictions.append({
                    "question": question[:100],
                    "gold": gold_answer,
                    "prediction": "ERROR",
                    "correct": False,
                    "error": str(e),
                })

        accuracy = correct / total if total > 0 else 0.0
        avg_time = float(np.mean(response_times)) if response_times else 0.0
        avg_halluc = float(np.mean(hallucination_rates)) if hallucination_rates else 0.0
        p95_time = float(np.percentile(response_times, 95)) if response_times else 0.0

        final_metrics = {
            "accuracy": accuracy,
            "hallucination_rate": avg_halluc,
            "avg_response_time_s": avg_time,
            "p95_response_time_s": p95_time,
            "total_questions": total,
            "correct_answers": correct,
        }
        mlflow.log_metrics(final_metrics)

        # Save predictions
        out = output_file or settings.data_processed_dir / f"medqa_predictions_{condition}.jsonl"
        with open(out, "w") as f:
            for p in predictions:
                f.write(json.dumps(p) + "\n")

        # Save latest metrics summary
        metrics_file = settings.data_processed_dir / "latest_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump({
                "experiment_name": settings.mlflow_experiment_name,
                **final_metrics,
                "last_updated": time.strftime("%Y-%m-%d %Human:%M:%S"),
            }, f, indent=2)

        logger.success(
            f"\n{'='*60}\n"
            f"Condition: {condition}\n"
            f"Accuracy: {accuracy:.4f} ({correct}/{total})\n"
            f"Hallucination Rate: {avg_halluc:.4f}\n"
            f"Avg Response Time: {avg_time:.2f}s (p95: {p95_time:.2f}s)\n"
            f"{'='*60}"
        )

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="MedQA accuracy evaluation")
    parser.add_argument("--condition", default="rag_with_hallucination_filter",
                        choices=["base_llm", "rag_only", "rag_with_hallucination_filter", "all"])
    parser.add_argument("--max-samples", type=int, default=settings.medqa_test_subset)
    args = parser.parse_args()

    settings.ensure_dirs()

    if args.condition == "all":
        for condition, use_rag, use_halluc in [
            ("base_llm", False, False),
            ("rag_only", True, False),
            ("rag_with_hallucination_filter", True, True),
        ]:
            run_medqa_evaluation(
                condition=condition,
                max_samples=args.max_samples,
                use_rag=use_rag,
                use_hallucination_filter=use_halluc,
            )
    else:
        use_rag = args.condition != "base_llm"
        use_halluc = args.condition == "rag_with_hallucination_filter"
        run_medqa_evaluation(
            condition=args.condition,
            max_samples=args.max_samples,
            use_rag=use_rag,
            use_hallucination_filter=use_halluc,
        )


if __name__ == "__main__":
    main()
