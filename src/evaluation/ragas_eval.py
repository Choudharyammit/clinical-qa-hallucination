"""
RAGAS-based faithfulness and answer relevance evaluation.

RAGAS metrics:
  - Faithfulness: fraction of answer claims supported by context
  - Answer Relevance: how well the answer addresses the question
  - Context Precision: signal-to-noise ratio of retrieved context
  - Context Recall: fraction of relevant info present in context

Usage:
    python -m src.evaluation.ragas_eval --max-samples 100
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
from datasets import Dataset
from loguru import logger

from src.config import settings


def run_ragas_evaluation(
    max_samples: int = 100,
    condition: str = "rag_with_hallucination_filter",
) -> dict:
    """
    Run RAGAS evaluation on a sample of PubMedQA questions.
    Returns dict of RAGAS metrics.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        logger.warning("RAGAS not installed. Run: pip install ragas")
        return {}

    from src.agents.orchestrator import OrchestratorAgent

    # Load PubMedQA labeled samples
    labeled_path = settings.data_raw_dir / "pubmedqa" / "labeled_train.jsonl"
    if not labeled_path.exists():
        logger.warning("PubMedQA labeled data not found. Run: make download-data first.")
        return {}

    samples = []
    with open(labeled_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    samples = samples[:max_samples]

    orchestrator = OrchestratorAgent()

    questions, answers, contexts_list, ground_truths = [], [], [], []

    logger.info(f"Running RAGAS evaluation on {len(samples)} PubMedQA samples…")

    for sample in samples:
        question = sample.get("question", "")
        gold_answer = sample.get("long_answer", "")
        evidence_contexts = sample.get("context", {}).get("contexts", [])

        if not question or not gold_answer:
            continue

        try:
            result = orchestrator.run(
                query=question,
                use_rag=True,
                use_hallucination_filter=True,
            )
            answer = result.get("answer", "")
        except Exception as e:
            logger.warning(f"RAGAS sample failed: {e}")
            continue

        questions.append(question)
        answers.append(answer)
        contexts_list.append(evidence_contexts[:5])  # RAGAS expects list[str] per question
        ground_truths.append(gold_answer)

    if not questions:
        logger.warning("No valid RAGAS samples collected.")
        return {}

    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    logger.info("Running RAGAS metrics computation…")
    try:
        ragas_result = evaluate(
            ragas_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )

        metrics = {
            "faithfulness": float(ragas_result.get("faithfulness", 0)),
            "answer_relevancy": float(ragas_result.get("answer_relevancy", 0)),
            "context_precision": float(ragas_result.get("context_precision", 0)),
            "context_recall": float(ragas_result.get("context_recall", 0)),
            "n_samples": len(questions),
        }

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        with mlflow.start_run(run_name=f"ragas_{condition}"):
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
            mlflow.log_param("condition", condition)
            mlflow.log_param("n_samples", len(questions))

        logger.success(
            f"\nRAGAS Results [{condition}]:\n"
            f"  Faithfulness:       {metrics['faithfulness']:.4f}\n"
            f"  Answer Relevancy:   {metrics['answer_relevancy']:.4f}\n"
            f"  Context Precision:  {metrics['context_precision']:.4f}\n"
            f"  Context Recall:     {metrics['context_recall']:.4f}"
        )
        return metrics

    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--condition", default="rag_with_hallucination_filter")
    args = parser.parse_args()
    settings.ensure_dirs()
    run_ragas_evaluation(args.max_samples, args.condition)


if __name__ == "__main__":
    main()
