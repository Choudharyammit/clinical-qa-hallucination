"""
A/B test harness for hallucination filter evaluation.

Runs identical MedQA questions through two conditions in parallel:
  - Condition A: RAG without hallucination filter
  - Condition B: RAG with hallucination filter (proposed system)

Computes accuracy, hallucination rate, and response time per condition.
Performs Welch's t-test for statistical significance on hallucination reduction.
Logs all metrics to MLflow.

Usage:
    from src.evaluation.ab_test import run_ab_test
    results = run_ab_test(questions, orchestrator, n=100)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import math
import numpy as np
from loguru import logger

from src.config import settings


@dataclass
class ABTestRecord:
    question_id: str
    question: str
    correct_answer: str
    mcq_options: dict[str, str]

    # Condition A: RAG without hallucination filter
    a_prediction: Optional[str] = None
    a_answer: str = ""
    a_hallucination_rate: float = 0.0
    a_response_time_s: float = 0.0
    a_correct: bool = False
    a_confidence: str = "Medium"

    # Condition B: RAG with hallucination filter (proposed)
    b_prediction: Optional[str] = None
    b_answer: str = ""
    b_hallucination_rate: float = 0.0
    b_response_time_s: float = 0.0
    b_correct: bool = False
    b_confidence: str = "Medium"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ABTestResults:
    n: int = 0

    # Condition A metrics
    a_accuracy: float = 0.0
    a_hallucination_mean: float = 0.0
    a_hallucination_std: float = 0.0
    a_avg_time_s: float = 0.0
    a_p95_time_s: float = 0.0

    # Condition B metrics
    b_accuracy: float = 0.0
    b_hallucination_mean: float = 0.0
    b_hallucination_std: float = 0.0
    b_avg_time_s: float = 0.0
    b_p95_time_s: float = 0.0

    # Comparison
    hallucination_reduction_pct: float = 0.0
    accuracy_delta: float = 0.0
    t_statistic: float = 0.0
    p_value: float = 1.0
    significant: bool = False  # p < 0.05

    records: list[ABTestRecord] = field(default_factory=list)

    def meets_50pct_target(self) -> bool:
        return self.hallucination_reduction_pct >= 50.0

    def print_summary(self) -> None:
        print(f"\n{'═'*65}")
        print("  A/B Test: Hallucination Filter Evaluation")
        print(f"{'═'*65}")
        print(f"  Questions evaluated: {self.n}")
        print(f"\n  {'Metric':<35} {'RAG Only':>10} {'Proposed':>10}")
        print(f"  {'─'*55}")
        print(f"  {'MedQA Accuracy':<35} {self.a_accuracy:>10.4f} {self.b_accuracy:>10.4f}")
        print(f"  {'Hallucination Rate (mean)':<35} {self.a_hallucination_mean:>10.4f} {self.b_hallucination_mean:>10.4f}")
        print(f"  {'Hallucination Rate (std)':<35} {self.a_hallucination_std:>10.4f} {self.b_hallucination_std:>10.4f}")
        print(f"  {'Avg Response Time (s)':<35} {self.a_avg_time_s:>10.2f} {self.b_avg_time_s:>10.2f}")
        print(f"  {'P95 Response Time (s)':<35} {self.a_p95_time_s:>10.2f} {self.b_p95_time_s:>10.2f}")
        print(f"\n  Hallucination reduction: {self.hallucination_reduction_pct:.1f}%  "
              f"{'✅ MEETS' if self.meets_50pct_target() else '❌ BELOW'} 50% target")
        print(f"  Welch t={self.t_statistic:.3f}, p={self.p_value:.4f}  "
              f"{'(significant)' if self.significant else '(not significant)'}")
        print(f"{'═'*65}\n")

    def to_dict(self) -> dict:
        d = {k: v for k, v in vars(self).items() if k != "records"}
        return d


def _welch_t_test(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's two-sample t-test. Returns (t_statistic, p_value)."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0, 1.0
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return 0.0, 1.0
    t = (mean_a - mean_b) / se
    # Welch–Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / denom if denom > 0 else 1.0
    # Approximate p-value using normal distribution for df > 30, else conservative
    p = _t_cdf_approx(abs(t), df)
    return t, p


def _t_cdf_approx(t: float, df: float) -> float:
    """Two-tailed p-value approximation without scipy."""
    # For large df this converges to standard normal; use erfc approximation
    x = df / (df + t * t)
    # Regularised incomplete beta function approximation via continued fraction
    # For thesis purposes: use normal approximation for df >= 30, else conservative estimate
    if df >= 30:
        # Normal distribution two-tailed: 2 * (1 - Phi(|t|))
        p = math.erfc(t / math.sqrt(2))
    else:
        # Conservative: use df=1 (Cauchy), overestimates p-value
        p = 1.0 - (2 / math.pi) * math.atan(t)
    return float(max(0.0, min(1.0, p)))


def run_ab_test(
    questions: list[dict],
    orchestrator,
    *,
    n: int = 100,
    output_dir: Optional[Path] = None,
    log_mlflow: bool = True,
) -> ABTestResults:
    """
    Run A/B test comparing RAG-only vs RAG+hallucination filter.

    Args:
        questions: List of MedQA dicts with 'question', 'answer', 'options' keys.
        orchestrator: OrchestratorAgent instance.
        n: Number of questions to evaluate (sampled from questions list).
        output_dir: Where to save per-question JSONL results.
        log_mlflow: Whether to log metrics to MLflow.

    Returns:
        ABTestResults with all metrics and per-question records.
    """
    import random
    rng = random.Random(settings.random_seed)
    sample = rng.sample(questions, min(n, len(questions)))

    records: list[ABTestRecord] = []
    total = len(sample)
    logger.info(f"A/B test: {total} questions × 2 conditions")

    for i, q in enumerate(sample):
        qid = q.get("question_id", str(i))
        question_text = q.get("question", "")
        correct = q.get("answer", "")
        options = q.get("options", {})

        rec = ABTestRecord(
            question_id=qid,
            question=question_text,
            correct_answer=correct,
            mcq_options=options,
        )

        # ── Condition A: RAG without hallucination filter ─────────────────────
        try:
            t0 = time.perf_counter()
            res_a = orchestrator.run(
                question_text,
                mcq_options=options,
                use_rag=True,
                use_hallucination_filter=False,
            )
            rec.a_response_time_s = time.perf_counter() - t0
            rec.a_prediction = res_a.get("mcq_prediction")
            rec.a_answer = res_a.get("answer", "")
            rec.a_hallucination_rate = float(res_a.get("hallucination_rate", 0.0))
            rec.a_confidence = res_a.get("confidence", "Medium")
            rec.a_correct = (rec.a_prediction == correct) if rec.a_prediction else False
        except Exception as e:
            logger.warning(f"[A/B] Condition A failed on q={qid}: {e}")

        # ── Condition B: RAG with hallucination filter ─────────────────────────
        try:
            t0 = time.perf_counter()
            res_b = orchestrator.run(
                question_text,
                mcq_options=options,
                use_rag=True,
                use_hallucination_filter=True,
            )
            rec.b_response_time_s = time.perf_counter() - t0
            rec.b_prediction = res_b.get("mcq_prediction")
            rec.b_answer = res_b.get("answer", "")
            rec.b_hallucination_rate = float(res_b.get("hallucination_rate", 0.0))
            rec.b_confidence = res_b.get("confidence", "Medium")
            rec.b_correct = (rec.b_prediction == correct) if rec.b_prediction else False
        except Exception as e:
            logger.warning(f"[A/B] Condition B failed on q={qid}: {e}")

        records.append(rec)
        if (i + 1) % 10 == 0 or (i + 1) == total:
            logger.info(f"  Progress: {i+1}/{total}")

    # ── Aggregate ──────────────────────────────────────────────────────────────
    a_rates = [r.a_hallucination_rate for r in records]
    b_rates = [r.b_hallucination_rate for r in records]
    a_times = [r.a_response_time_s for r in records]
    b_times = [r.b_response_time_s for r in records]

    a_acc = float(np.mean([r.a_correct for r in records])) if records else 0.0
    b_acc = float(np.mean([r.b_correct for r in records])) if records else 0.0
    base_h = float(np.mean(a_rates)) if a_rates else 0.0
    prop_h = float(np.mean(b_rates)) if b_rates else 0.0
    reduction = (base_h - prop_h) / base_h * 100.0 if base_h > 0 else 0.0

    t_stat, p_val = _welch_t_test(a_rates, b_rates)

    results = ABTestResults(
        n=len(records),
        a_accuracy=a_acc,
        a_hallucination_mean=base_h,
        a_hallucination_std=float(np.std(a_rates)) if a_rates else 0.0,
        a_avg_time_s=float(np.mean(a_times)) if a_times else 0.0,
        a_p95_time_s=float(np.percentile(a_times, 95)) if a_times else 0.0,
        b_accuracy=b_acc,
        b_hallucination_mean=prop_h,
        b_hallucination_std=float(np.std(b_rates)) if b_rates else 0.0,
        b_avg_time_s=float(np.mean(b_times)) if b_times else 0.0,
        b_p95_time_s=float(np.percentile(b_times, 95)) if b_times else 0.0,
        hallucination_reduction_pct=reduction,
        accuracy_delta=b_acc - a_acc,
        t_statistic=t_stat,
        p_value=p_val,
        significant=(p_val < 0.05),
        records=records,
    )

    # ── Save per-question JSONL ────────────────────────────────────────────────
    out_dir = output_dir or settings.data_processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "ab_test_records.jsonl"
    with open(records_path, "w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")
    logger.info(f"Saved {len(records)} A/B records → {records_path}")

    summary_path = out_dir / "ab_test_summary.json"
    summary_path.write_text(json.dumps(results.to_dict(), indent=2))
    logger.info(f"Saved summary → {summary_path}")

    # ── MLflow logging ─────────────────────────────────────────────────────────
    if log_mlflow:
        _log_ab_test_to_mlflow(results)

    return results


def _log_ab_test_to_mlflow(results: ABTestResults) -> None:
    """Log A/B test results to MLflow as two separate condition runs."""
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

        # Condition A
        with mlflow.start_run(run_name="rag_only"):
            mlflow.log_metrics({
                "accuracy": results.a_accuracy,
                "hallucination_rate": results.a_hallucination_mean,
                "hallucination_std": results.a_hallucination_std,
                "avg_response_time_s": results.a_avg_time_s,
                "p95_response_time_s": results.a_p95_time_s,
                "n_questions": results.n,
            })
            mlflow.log_params({
                "use_rag": True,
                "use_hallucination_filter": False,
                "condition": "rag_only",
            })

        # Condition B
        with mlflow.start_run(run_name="rag_with_hallucination_filter"):
            mlflow.log_metrics({
                "accuracy": results.b_accuracy,
                "hallucination_rate": results.b_hallucination_mean,
                "hallucination_std": results.b_hallucination_std,
                "avg_response_time_s": results.b_avg_time_s,
                "p95_response_time_s": results.b_p95_time_s,
                "hallucination_reduction_pct": results.hallucination_reduction_pct,
                "accuracy_delta_vs_rag_only": results.accuracy_delta,
                "welch_t_statistic": results.t_statistic,
                "welch_p_value": results.p_value,
                "n_questions": results.n,
            })
            mlflow.log_params({
                "use_rag": True,
                "use_hallucination_filter": True,
                "condition": "rag_with_hallucination_filter",
                "statistical_significance": str(results.significant),
            })

        logger.success("A/B test results logged to MLflow.")

    except Exception as e:
        logger.warning(f"MLflow logging failed (non-fatal): {e}")


def load_ab_test_results(
    path: Optional[Path] = None,
) -> Optional[ABTestResults]:
    """Load saved A/B test summary from JSON."""
    p = path or (settings.data_processed_dir / "ab_test_summary.json")
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    r = ABTestResults()
    for k, v in d.items():
        if hasattr(r, k):
            setattr(r, k, v)
    return r
