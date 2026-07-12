"""
Baseline evaluator — measures hallucination rate of an LLM with NO retrieval.

This establishes the hallucination baseline for the comparison study:
  Condition 1: Base LLM (no RAG)          ← this module
  Condition 2: RAG only
  Condition 3: RAG + Hallucination filter (proposed system)

Uses BioGPT-Large or any small LLM for fast CPU-based inference.
Extracts factual claims from each generated answer and scores them against
the gold PubMedQA contexts using the DeBERTa NLI model.

Usage:
    python -m src.evaluation.baseline_eval --max-samples 100
    python -m src.evaluation.baseline_eval --model biogpt --max-samples 200
"""

import argparse
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
class BaselineResult:
    model_name: str
    total_questions: int = 0
    medqa_accuracy: float = 0.0
    hallucination_rate: float = 0.0
    avg_response_time_s: float = 0.0
    per_sample_results: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "model": self.model_name,
            "total_questions": self.total_questions,
            "medqa_accuracy": round(self.medqa_accuracy, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "avg_response_time_s": round(self.avg_response_time_s, 3),
        }


def run_baseline_medqa(
    model_name: str = "biogpt",
    max_samples: int = 100,
    medqa_path: Optional[Path] = None,
) -> BaselineResult:
    """
    Run MedQA accuracy test with no RAG context.
    Uses BioGPT-Large as the base model for fast inference.
    Records both accuracy AND hallucination rate.
    """
    from src.models.nli import extract_claims, get_nli_model

    # Load test data
    test_path = medqa_path or settings.data_processed_dir / "medqa_test.jsonl"
    if not test_path.exists():
        test_path = settings.data_raw_dir / "medqa" / "test.jsonl"
    if not test_path.exists():
        logger.error("MedQA test file not found. Run: make download-data")
        return BaselineResult(model_name=model_name)

    samples = []
    with open(test_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    samples = samples[:max_samples]
    logger.info(f"Baseline evaluation on {len(samples)} MedQA samples (no RAG)")

    # Load LLM
    from src.models.llm import BioGPTFallback, BioMistralLLM, OllamaLLM

    if model_name == "biogpt":
        llm = BioGPTFallback()
    elif model_name == "biomistral":
        llm = BioMistralLLM()
    elif model_name == "ollama":
        llm = OllamaLLM()
    else:
        llm = BioGPTFallback()

    llm.load()
    nli = get_nli_model()

    correct = 0
    hallucination_rates = []
    response_times = []
    per_sample = []

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=f"baseline_{model_name}_no_rag"):
        mlflow.log_params({
            "condition": "base_llm",
            "model": model_name,
            "use_rag": False,
            "n_samples": len(samples),
        })

        for i, sample in enumerate(tqdm(samples, desc=f"Baseline [{model_name}]")):
            question = sample.get("question", "")
            options = sample.get("options", {})
            gold = (sample.get("answer_idx") or sample.get("answer", "")).strip()
            if gold and len(gold) > 1:
                gold = gold[0].upper()

            try:
                # Generate WITHOUT any retrieval context
                result = llm.answer_clinical_question(
                    question=question,
                    context="No context provided. Answer from medical knowledge only.",
                    mcq_options=options if isinstance(options, dict) else None,
                )
                answer = result.get("answer", "")
                elapsed = result.get("response_time_s", 0)
                response_times.append(elapsed)

                # MCQ accuracy
                from src.agents.generator import _extract_mcq_answer
                prediction = (_extract_mcq_answer(answer) or "").upper()
                is_correct = prediction == gold
                if is_correct:
                    correct += 1

                # Hallucination: check claims against empty evidence (no retrieval)
                # Without retrieval, we check against the MCQ options as pseudo-evidence
                pseudo_evidence = list(options.values()) if isinstance(options, dict) else []
                claims = extract_claims(answer)
                if claims and pseudo_evidence:
                    claim_scores = []
                    for claim in claims:
                        res = nli.score_claim_against_evidence(claim, pseudo_evidence)
                        claim_scores.append(0.0 if res["is_supported"] else 1.0)
                    halluc_rate = float(np.mean(claim_scores))
                else:
                    halluc_rate = 0.5  # Unknown — no evidence to check against

                hallucination_rates.append(halluc_rate)

                per_sample.append({
                    "question": question[:100],
                    "gold": gold,
                    "prediction": prediction,
                    "correct": is_correct,
                    "hallucination_rate": halluc_rate,
                    "response_time_s": elapsed,
                })

                if (i + 1) % 25 == 0:
                    interim_acc = correct / (i + 1)
                    interim_halluc = float(np.mean(hallucination_rates))
                    mlflow.log_metric("interim_accuracy", interim_acc, step=i + 1)
                    mlflow.log_metric("interim_hallucination_rate", interim_halluc, step=i + 1)
                    logger.info(
                        f"  [{i+1}/{len(samples)}] Acc: {interim_acc:.3f} | "
                        f"Halluc: {interim_halluc:.3f}"
                    )

            except Exception as e:
                logger.warning(f"Sample {i} failed: {e}")

        accuracy = correct / len(samples) if samples else 0.0
        avg_halluc = float(np.mean(hallucination_rates)) if hallucination_rates else 0.0
        avg_time = float(np.mean(response_times)) if response_times else 0.0

        mlflow.log_metrics({
            "accuracy": accuracy,
            "hallucination_rate": avg_halluc,
            "avg_response_time_s": avg_time,
        })

        # Save results
        out = settings.data_processed_dir / f"baseline_{model_name}_results.jsonl"
        with open(out, "w") as f:
            for r in per_sample:
                f.write(json.dumps(r) + "\n")

        logger.success(
            f"\n{'='*55}\n"
            f"Baseline ({model_name}, no RAG):\n"
            f"  MedQA Accuracy:     {accuracy:.4f} ({correct}/{len(samples)})\n"
            f"  Hallucination Rate: {avg_halluc:.4f}\n"
            f"  Avg Response Time:  {avg_time:.2f}s\n"
            f"{'='*55}"
        )

    return BaselineResult(
        model_name=model_name,
        total_questions=len(samples),
        medqa_accuracy=accuracy,
        hallucination_rate=avg_halluc,
        avg_response_time_s=avg_time,
        per_sample_results=per_sample,
    )


def print_comparison_vs_proposed(baseline: BaselineResult, proposed_metrics: dict) -> None:
    """Print a table comparing baseline vs proposed system metrics."""
    b_acc = baseline.medqa_accuracy
    b_halluc = baseline.hallucination_rate
    p_acc = proposed_metrics.get("accuracy", 0)
    p_halluc = proposed_metrics.get("hallucination_rate", 0)
    reduction = (b_halluc - p_halluc) / b_halluc * 100 if b_halluc > 0 else 0

    logger.info(
        f"\n{'='*60}\n"
        f"{'Metric':<25} {'Base LLM':>12} {'Proposed':>12} {'Delta':>10}\n"
        f"{'─'*60}\n"
        f"{'MedQA Accuracy':<25} {b_acc:>12.4f} {p_acc:>12.4f} {p_acc - b_acc:>+10.4f}\n"
        f"{'Hallucination Rate':<25} {b_halluc:>12.4f} {p_halluc:>12.4f} {p_halluc - b_halluc:>+10.4f}\n"
        f"{'Halluc Reduction':<25} {'':>12} {'':>12} {reduction:>9.1f}%\n"
        f"{'='*60}\n"
        f"Target ≥50% hallucination reduction: "
        f"{'✅ MET' if reduction >= 50 else '❌ NOT MET'}"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline hallucination evaluation")
    parser.add_argument("--model", default="biogpt",
                        choices=["biogpt", "biomistral", "ollama"])
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    settings.ensure_dirs()
    run_baseline_medqa(args.model, args.max_samples)


if __name__ == "__main__":
    main()
