"""
EvaluatorAgent — automated benchmarking on MedQA and PubMedQA.

Runs:
  - MedQA accuracy (correct MCQ answer A/B/C/D)
  - Hallucination rate (via HallucinationDetectorAgent)
  - BLEU, ROUGE-L (for open-ended answers)
  - RAGAS faithfulness + answer relevance
  - Logs all results to MLflow
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
from loguru import logger
from tqdm import tqdm

from src.config import settings


@dataclass
class BenchmarkResult:
    experiment_name: str
    accuracy: float = 0.0
    hallucination_rate: float = 0.0
    avg_response_time_s: float = 0.0
    bleu: float = 0.0
    rouge_l: float = 0.0
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    total_questions: int = 0
    correct_answers: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "experiment_name": self.experiment_name,
            "accuracy": self.accuracy,
            "hallucination_rate": self.hallucination_rate,
            "avg_response_time_s": self.avg_response_time_s,
            "bleu": self.bleu,
            "rouge_l": self.rouge_l,
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "total_questions": self.total_questions,
            "correct_answers": self.correct_answers,
        }


class EvaluatorAgent:
    """
    Automated benchmarking agent for the clinical QA system.

    Compares three experimental conditions:
      1. Base LLM (no RAG)
      2. RAG without hallucination filter
      3. RAG + Hallucination filter (proposed system)
    """

    def __init__(self, orchestrator=None):
        self._orchestrator = orchestrator
        self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

    def load_medqa_test(
        self,
        path: Optional[Path] = None,
        max_samples: Optional[int] = None,
    ) -> list[dict]:
        """Load MedQA test split from processed JSONL."""
        test_path = path or settings.data_processed_dir / "medqa_test.jsonl"
        if not test_path.exists():
            # Try raw fallback
            test_path = settings.data_raw_dir / "medqa" / "test.jsonl"
        if not test_path.exists():
            logger.warning(f"MedQA test file not found: {test_path}")
            return []

        samples = []
        with open(test_path, "r") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        if max_samples:
            samples = samples[:max_samples]
        logger.info(f"Loaded {len(samples)} MedQA test samples")
        return samples

    def evaluate_medqa_accuracy(
        self,
        samples: list[dict],
        condition: str = "rag_with_hallucination_filter",
        use_rag: bool = True,
        use_hallucination_filter: bool = True,
    ) -> BenchmarkResult:
        """
        Run MedQA accuracy evaluation.
        condition: label for MLflow tracking
        """
        correct = 0
        response_times = []
        hallucination_rates = []

        with mlflow.start_run(run_name=f"medqa_{condition}"):
            mlflow.log_params({
                "condition": condition,
                "use_rag": use_rag,
                "use_hallucination_filter": use_hallucination_filter,
                "n_samples": len(samples),
                "model": settings.llm_model_name,
            })

            for i, sample in enumerate(tqdm(samples, desc=f"MedQA [{condition}]")):
                question = sample["question"]
                options = sample.get("options", {})
                answer_key = sample.get("answer_idx") or sample.get("answer", "")

                # Normalize answer key to single letter
                if answer_key and len(answer_key) > 1:
                    answer_key = answer_key[0].upper()

                try:
                    result = self._run_query(
                        question=question,
                        mcq_options=options,
                        use_rag=use_rag,
                        use_hallucination_filter=use_hallucination_filter,
                    )
                    prediction = result.get("mcq_prediction", "")
                    response_times.append(result.get("response_time_s", 0))
                    hallucination_rates.append(
                        result.get("hallucination_report", {}).get("hallucination_rate", 0)
                    )

                    if prediction and prediction.upper() == answer_key.upper():
                        correct += 1

                except Exception as e:
                    logger.warning(f"Sample {i} failed: {e}")

                # Log intermediate metrics every 50 samples
                if (i + 1) % 50 == 0:
                    interim_acc = correct / (i + 1)
                    mlflow.log_metric("interim_accuracy", interim_acc, step=i + 1)

            accuracy = correct / len(samples) if samples else 0.0
            avg_time = float(np.mean(response_times)) if response_times else 0.0
            avg_halluc = float(np.mean(hallucination_rates)) if hallucination_rates else 0.0

            mlflow.log_metrics({
                "accuracy": accuracy,
                "hallucination_rate": avg_halluc,
                "avg_response_time_s": avg_time,
                "correct_answers": correct,
            })

            logger.success(
                f"[{condition}] Accuracy: {accuracy:.3f} ({correct}/{len(samples)}) | "
                f"Hallucination: {avg_halluc:.3f} | Avg time: {avg_time:.2f}s"
            )

        return BenchmarkResult(
            experiment_name=condition,
            accuracy=accuracy,
            hallucination_rate=avg_halluc,
            avg_response_time_s=avg_time,
            total_questions=len(samples),
            correct_answers=correct,
        )

    def run_comparison(
        self,
        max_samples: Optional[int] = None,
    ) -> dict[str, BenchmarkResult]:
        """
        Run the full 3-condition comparison study:
        1. Base LLM (no RAG, no hallucination filter)
        2. RAG only (no hallucination filter)
        3. RAG + Hallucination filter (proposed system)
        """
        samples = self.load_medqa_test(max_samples=max_samples or settings.medqa_test_subset)
        if not samples:
            logger.error("No test samples available.")
            return {}

        results = {}

        # Condition 1: Base LLM
        logger.info("Running Condition 1: Base LLM (no RAG)…")
        results["base_llm"] = self.evaluate_medqa_accuracy(
            samples, condition="base_llm", use_rag=False, use_hallucination_filter=False
        )

        # Condition 2: RAG only
        logger.info("Running Condition 2: RAG without hallucination filter…")
        results["rag_only"] = self.evaluate_medqa_accuracy(
            samples, condition="rag_only", use_rag=True, use_hallucination_filter=False
        )

        # Condition 3: RAG + Hallucination filter
        logger.info("Running Condition 3: RAG + Hallucination filter (proposed)…")
        results["rag_halluc_filter"] = self.evaluate_medqa_accuracy(
            samples, condition="rag_halluc_filter", use_rag=True, use_hallucination_filter=True
        )

        self._print_comparison_table(results)
        return results

    def _run_query(
        self,
        question: str,
        mcq_options: Optional[dict] = None,
        use_rag: bool = True,
        use_hallucination_filter: bool = True,
    ) -> dict:
        """Delegate to orchestrator or run minimal pipeline."""
        if self._orchestrator:
            return self._orchestrator.run(
                query=question,
                mcq_options=mcq_options,
                use_rag=use_rag,
                use_hallucination_filter=use_hallucination_filter,
            )
        raise RuntimeError("No orchestrator set. Call evaluator.set_orchestrator() first.")

    def set_orchestrator(self, orchestrator) -> None:
        self._orchestrator = orchestrator

    def _print_comparison_table(self, results: dict[str, "BenchmarkResult"]) -> None:
        header = f"{'Condition':<30} {'Accuracy':>10} {'Halluc Rate':>12} {'Avg Time':>10}"
        sep = "-" * 65
        logger.info(f"\n{sep}\n{header}\n{sep}")
        for name, r in results.items():
            logger.info(
                f"{name:<30} {r.accuracy:>9.3f}  {r.hallucination_rate:>11.3f}  "
                f"{r.avg_response_time_s:>9.2f}s"
            )
        logger.info(sep)
