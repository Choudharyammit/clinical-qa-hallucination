#!/usr/bin/env python3
"""
Four-Condition Hallucination Evaluation — LJMU MSc Dissertation
================================================================
Conditions:
  A: Base LLM          — BioGPT-347M, no retrieval, no NLI filter
  B: RAG (embed only)  — BioBERT retrieval, NO cross-encoder reranker, no filter
  C: RAG + Reranker    — BioBERT + TinyBERT reranker, no NLI filter
  D: Full System       — RAG + Reranker + NLI filter (proposed system)

Hallucination Measurement (UNIFORM across all conditions):
  For each generated answer, split into factual claims (sentences).
  Check each claim against the PubMedQA gold evidence abstract using NLI.
  Hallucination rate = fraction of claims NOT entailed by gold evidence.

  Using gold evidence as the universal reference ensures condition A
  (no retrieval) is measured on the same scale as conditions B/C/D.
  This is the methodology recommended in RAGChecker (Ru et al. 2024).

Statistical Tests:
  Welch's two-sample t-test (H0: equal hallucination means):
  - A vs D: primary hypothesis H1-1 (target ≥50% relative reduction)
  - C vs D: reranking ablation H1-3
  - A vs C: retrieval effect

Research hypotheses tested:
  H1-1: ≥50% relative hallucination reduction (A→D)
  H1-3: Cross-encoder reranking reduces hallucination (B→C)

Usage:
  cd clinical-qa-system
  python scripts/run_four_condition_eval.py --n 50 --output results/four_condition_eval.json
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import numpy as np
from loguru import logger

from src.config import settings
from src.models.llm import BioGPTFallback
from src.models.nli import MedNLIModel, extract_claims
from src.agents.retriever import RetrieverAgent
from src.agents.reranker import RerankAgent
from src.agents.hallucination import HallucinationDetectorAgent


# ── Constants ──────────────────────────────────────────────────────────────────

PUBMEDQA_PATH = Path("data/processed/pubmedqa_labeled_train_chunks.jsonl")
RESULTS_DIR = Path("results")
MEASUREMENT_THRESHOLD = 0.35  # Lower threshold for lenient measurement

# Yes/no/maybe answer patterns for BioGPT output
YES_PATTERNS = ("yes", "true", "positive", "confirm", "support", "likely", "do ")
NO_PATTERNS = ("no", "false", "negative", "not ", "unlikely", "neither", "don't")
MAYBE_PATTERNS = ("maybe", "possibly", "uncertain", "unclear", "might", "could",
                  "inconclusive", "limited")


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_pubmedqa_questions(path: Path, n: int = 50, seed: int = 42) -> list[dict]:
    """
    Load PubMedQA questions with gold evidence, deduplicated by pubid.
    Returns n questions stratified by final_decision (yes/no/maybe).
    """
    questions_by_id: dict[int, dict] = {}

    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            pid = d["pubid"]
            if pid not in questions_by_id:
                questions_by_id[pid] = {
                    "pubid": pid,
                    "question": d["question"],
                    "final_decision": d["final_decision"],
                    "gold_evidence": [],
                }
            text = d.get("text", "").strip()
            if text:
                questions_by_id[pid]["gold_evidence"].append(text)

    # Filter questions that have gold evidence
    all_q = [q for q in questions_by_id.values() if q["gold_evidence"]]

    # Stratified sample: balanced yes/no/maybe
    rng = __import__("random").Random(seed)
    by_decision: dict[str, list] = defaultdict(list)
    for q in all_q:
        by_decision[q["final_decision"]].append(q)

    per_class = max(1, n // 3)
    sampled: list[dict] = []
    for decision in ["yes", "no", "maybe"]:
        pool = by_decision.get(decision, [])
        rng.shuffle(pool)
        sampled.extend(pool[:per_class])

    # Fill remaining slots if one class has fewer samples
    rng.shuffle(sampled)
    if len(sampled) < n:
        remaining = [q for q in all_q if q not in sampled]
        rng.shuffle(remaining)
        sampled.extend(remaining[: n - len(sampled)])

    result = sampled[:n]
    logger.info(
        f"Loaded {len(result)} PubMedQA questions | "
        f"yes={sum(1 for q in result if q['final_decision']=='yes')}, "
        f"no={sum(1 for q in result if q['final_decision']=='no')}, "
        f"maybe={sum(1 for q in result if q['final_decision']=='maybe')}"
    )
    return result


# ── Generation ─────────────────────────────────────────────────────────────────

def format_context(docs, max_chars: int = 1500) -> str:
    """Format retrieved documents into a context string."""
    parts = []
    total = 0
    for doc in docs:
        text = getattr(doc, "text", doc.get("text", "") if isinstance(doc, dict) else "")
        if total + len(text) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                parts.append(text[:remaining])
            break
        parts.append(text)
        total += len(text)
    return "\n\n".join(parts)


def generate_cond_a(question: str, llm: BioGPTFallback) -> tuple[str, float]:
    """Condition A: Base LLM only — no context, no filter."""
    t0 = time.perf_counter()
    prompt = f"Question: {question}\nAnswer:"
    answer = llm.generate(prompt, max_new_tokens=200)
    return answer, time.perf_counter() - t0


def generate_cond_b(
    question: str,
    llm: BioGPTFallback,
    retriever: RetrieverAgent,
) -> tuple[str, float, list]:
    """Condition B: RAG with embedding similarity only — no cross-encoder reranker."""
    t0 = time.perf_counter()
    docs = retriever.retrieve(question, top_k=10)
    context = format_context(docs[:5])
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    answer = llm.generate(prompt, max_new_tokens=200)
    return answer, time.perf_counter() - t0, docs[:5]


def generate_cond_c(
    question: str,
    llm: BioGPTFallback,
    retriever: RetrieverAgent,
    reranker: RerankAgent,
) -> tuple[str, float, list]:
    """Condition C: RAG + cross-encoder reranker — no NLI filter."""
    t0 = time.perf_counter()
    docs = retriever.retrieve(question, top_k=10)
    reranked = reranker.rerank(question, docs, top_k=5)
    context = format_context(reranked)
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    answer = llm.generate(prompt, max_new_tokens=200)
    return answer, time.perf_counter() - t0, reranked


def generate_cond_d(
    question: str,
    llm: BioGPTFallback,
    retriever: RetrieverAgent,
    reranker: RerankAgent,
    halluc_detector: HallucinationDetectorAgent,
) -> tuple[str, float, list, float]:
    """
    Condition D: Full system — RAG + reranker + NLI filter.
    NLI filter removes claims not grounded in retrieved evidence.
    Returns (filtered_answer, latency, reranked_docs, pipeline_halluc_rate).
    """
    t0 = time.perf_counter()
    docs = retriever.retrieve(question, top_k=10)
    reranked = reranker.rerank(question, docs, top_k=5)
    context = format_context(reranked)

    # Generate initial answer
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    raw_answer = llm.generate(prompt, max_new_tokens=200)

    # Apply NLI filter: remove claims not entailed by retrieved evidence
    evidence_texts = [
        (getattr(d, "text", d.get("text", "") if isinstance(d, dict) else ""))
        for d in reranked
    ]
    evidence_texts = [t for t in evidence_texts if t]
    report = halluc_detector.verify(raw_answer, evidence_texts)
    # Use empty string when all claims are removed (do NOT fall back to raw_answer —
    # that would mask the filter's effect). Empty string = "no verified claims remain."
    filtered_answer = report.verified_response.strip() if report.verified_response else ""

    return filtered_answer, time.perf_counter() - t0, reranked, report.hallucination_rate


# ── Measurement ────────────────────────────────────────────────────────────────

def measure_hallucination_vs_gold(
    answer: str,
    gold_evidence: list[str],
    nli: MedNLIModel,
    threshold: float = MEASUREMENT_THRESHOLD,
) -> dict:
    """
    Measure hallucination rate of an answer against gold evidence.

    For each claim in the answer, check entailment against gold evidence chunks.
    Hallucination rate = fraction of claims with max_entailment < threshold.
    Mean entailment score is also reported as a continuous measure.

    Returns dict with: hallucination_rate, mean_entailment, n_claims, claim_scores.
    """
    claims = extract_claims(answer)
    if not claims:
        return {
            "hallucination_rate": 0.0,
            "mean_entailment": 0.0,
            "n_claims": 0,
            "claim_scores": [],
        }

    claim_scores = []
    for claim in claims:
        result = nli.score_claim_against_evidence(claim, gold_evidence)
        claim_scores.append(result["max_entailment"])

    halluc_rate = sum(1 for s in claim_scores if s < threshold) / len(claim_scores)
    mean_ent = float(np.mean(claim_scores))

    return {
        "hallucination_rate": round(halluc_rate, 4),
        "mean_entailment": round(mean_ent, 4),
        "n_claims": len(claims),
        "claim_scores": [round(s, 4) for s in claim_scores],
    }


def classify_answer(answer: str) -> str:
    """Classify generated answer as yes/no/maybe based on text patterns."""
    a = answer.lower()[:200]
    for p in YES_PATTERNS:
        if p in a:
            return "yes"
    for p in NO_PATTERNS:
        if p in a:
            return "no"
    for p in MAYBE_PATTERNS:
        if p in a:
            return "maybe"
    return "unknown"


# ── Statistics ──────────────────────────────────────────────────────────────────

def welch_t_test(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's two-sample t-test. Returns (t_statistic, two_tailed_p_value)."""
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
    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = ((var_a / n_a) ** 2 / (n_a - 1)) + ((var_b / n_b) ** 2 / (n_b - 1))
    df = num / denom if denom > 0 else 1.0
    # Two-tailed p-value (normal approximation for df >= 30)
    if df >= 30:
        p = float(math.erfc(abs(t) / math.sqrt(2)))
    else:
        p = float(1.0 - (2 / math.pi) * math.atan(abs(t)))
    return round(t, 4), round(max(0.0, min(1.0, p)), 6)


def cohen_d(a: list[float], b: list[float]) -> float:
    """Cohen's d effect size."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled_std = math.sqrt(
        (np.var(a, ddof=1) * (len(a) - 1) + np.var(b, ddof=1) * (len(b) - 1))
        / (len(a) + len(b) - 2)
    )
    if pooled_std == 0:
        return 0.0
    return round((float(np.mean(a)) - float(np.mean(b))) / pooled_std, 4)


# ── MLflow Logging ─────────────────────────────────────────────────────────────

def log_to_mlflow(summary: dict) -> None:
    """Log four-condition results to MLflow."""
    try:
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

        conditions = ["A", "B", "C", "D"]
        cond_names = {
            "A": "base_llm_no_rag",
            "B": "rag_embed_only",
            "C": "rag_plus_reranker",
            "D": "full_system_proposed",
        }

        for cond in conditions:
            stats = summary["condition_stats"].get(cond, {})
            if not stats:
                continue
            with mlflow.start_run(run_name=cond_names[cond]):
                mlflow.log_params({
                    "condition": cond,
                    "condition_name": cond_names[cond],
                    "n_questions": summary["n_questions"],
                    "nli_model": summary["nli_model"],
                    "llm_model": summary["llm_model"],
                    "measurement_threshold": summary["measurement_threshold"],
                })
                mlflow.log_metrics({
                    "hallucination_rate_mean": stats["hallucination_rate_mean"],
                    "hallucination_rate_std": stats["hallucination_rate_std"],
                    "mean_entailment_score": stats["mean_entailment_mean"],
                    "avg_latency_s": stats["avg_latency_s"],
                    "p95_latency_s": stats["p95_latency_s"],
                    "n_claims_per_answer": stats["avg_n_claims"],
                })

        # Log comparison metrics in a separate run
        with mlflow.start_run(run_name="four_condition_comparison"):
            cmp = summary.get("comparisons", {})
            mlflow.log_params({
                "experiment": "four_condition_hallucination_eval",
                "n_questions": summary["n_questions"],
            })
            for key, val in cmp.items():
                if isinstance(val, (int, float)):
                    mlflow.log_metric(key.replace(" ", "_").replace("→", "_to_"), val)

        logger.success("Results logged to MLflow.")
    except Exception as e:
        logger.warning(f"MLflow logging failed (non-fatal): {e}")


# ── Main Evaluation ────────────────────────────────────────────────────────────

def run_evaluation(n: int = 50, output_path: Path | None = None) -> dict:
    logger.info("=" * 70)
    logger.info("Four-Condition Hallucination Evaluation")
    logger.info(f"Questions: {n} | NLI threshold: {MEASUREMENT_THRESHOLD}")
    logger.info("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────────────
    questions = load_pubmedqa_questions(PUBMEDQA_PATH, n=n)

    # ── Load models ────────────────────────────────────────────────────────────
    logger.info("Loading models...")

    retriever = RetrieverAgent()
    reranker = RerankAgent()

    llm = BioGPTFallback()
    llm.load()

    nli = MedNLIModel(threshold=MEASUREMENT_THRESHOLD)
    nli.load()

    halluc_detector = HallucinationDetectorAgent(
        nli_model=nli,
        flag_mode="remove",
    )

    logger.info("All models loaded. Starting evaluation...")

    # ── Per-question results ───────────────────────────────────────────────────
    records: list[dict] = []
    total = len(questions)

    for i, q in enumerate(questions):
        qid = q["pubid"]
        question_text = q["question"]
        gold_evidence = q["gold_evidence"]
        gold_decision = q["final_decision"]

        logger.info(f"[{i+1}/{total}] Q{qid}: {question_text[:60]}...")
        rec: dict = {
            "pubid": qid,
            "question": question_text,
            "gold_decision": gold_decision,
            "gold_evidence_chunks": len(gold_evidence),
        }

        # ── Condition A: Base LLM ──────────────────────────────────────────────
        try:
            ans_a, lat_a = generate_cond_a(question_text, llm)
            meas_a = measure_hallucination_vs_gold(ans_a, gold_evidence, nli)
            pred_a = classify_answer(ans_a)
            rec["A"] = {
                "answer": ans_a[:400],
                "latency_s": round(lat_a, 3),
                "predicted_decision": pred_a,
                "correct": pred_a == gold_decision,
                **meas_a,
            }
            logger.debug(
                f"  A: halluc={meas_a['hallucination_rate']:.2f}  "
                f"ent={meas_a['mean_entailment']:.3f}  lat={lat_a:.1f}s"
            )
        except Exception as e:
            logger.warning(f"  Condition A failed: {e}")
            rec["A"] = {"hallucination_rate": 1.0, "mean_entailment": 0.0,
                        "latency_s": 0.0, "n_claims": 0, "claim_scores": [],
                        "answer": "", "predicted_decision": "unknown", "correct": False}

        # ── Condition B: RAG embedding only ────────────────────────────────────
        try:
            ans_b, lat_b, docs_b = generate_cond_b(question_text, llm, retriever)
            meas_b = measure_hallucination_vs_gold(ans_b, gold_evidence, nli)
            pred_b = classify_answer(ans_b)
            rec["B"] = {
                "answer": ans_b[:400],
                "latency_s": round(lat_b, 3),
                "predicted_decision": pred_b,
                "correct": pred_b == gold_decision,
                "docs_retrieved": len(docs_b),
                **meas_b,
            }
            logger.debug(
                f"  B: halluc={meas_b['hallucination_rate']:.2f}  "
                f"ent={meas_b['mean_entailment']:.3f}  lat={lat_b:.1f}s"
            )
        except Exception as e:
            logger.warning(f"  Condition B failed: {e}")
            rec["B"] = {"hallucination_rate": 1.0, "mean_entailment": 0.0,
                        "latency_s": 0.0, "n_claims": 0, "claim_scores": [],
                        "answer": "", "predicted_decision": "unknown", "correct": False,
                        "docs_retrieved": 0}

        # ── Condition C: RAG + reranker ────────────────────────────────────────
        try:
            ans_c, lat_c, docs_c = generate_cond_c(question_text, llm, retriever, reranker)
            meas_c = measure_hallucination_vs_gold(ans_c, gold_evidence, nli)
            pred_c = classify_answer(ans_c)
            rec["C"] = {
                "answer": ans_c[:400],
                "latency_s": round(lat_c, 3),
                "predicted_decision": pred_c,
                "correct": pred_c == gold_decision,
                "docs_retrieved": len(docs_c),
                **meas_c,
            }
            logger.debug(
                f"  C: halluc={meas_c['hallucination_rate']:.2f}  "
                f"ent={meas_c['mean_entailment']:.3f}  lat={lat_c:.1f}s"
            )
        except Exception as e:
            logger.warning(f"  Condition C failed: {e}")
            rec["C"] = {"hallucination_rate": 1.0, "mean_entailment": 0.0,
                        "latency_s": 0.0, "n_claims": 0, "claim_scores": [],
                        "answer": "", "predicted_decision": "unknown", "correct": False,
                        "docs_retrieved": 0}

        # ── Condition D: Full system ───────────────────────────────────────────
        try:
            ans_d, lat_d, docs_d, pipe_halluc = generate_cond_d(
                question_text, llm, retriever, reranker, halluc_detector
            )
            meas_d = measure_hallucination_vs_gold(ans_d, gold_evidence, nli)
            pred_d = classify_answer(ans_d)
            rec["D"] = {
                "answer": ans_d[:400],
                "latency_s": round(lat_d, 3),
                "predicted_decision": pred_d,
                "correct": pred_d == gold_decision,
                "docs_retrieved": len(docs_d),
                "pipeline_hallucination_rate": round(pipe_halluc, 4),
                **meas_d,
            }
            logger.debug(
                f"  D: halluc={meas_d['hallucination_rate']:.2f}  "
                f"ent={meas_d['mean_entailment']:.3f}  lat={lat_d:.1f}s"
                f"  (pipe_halluc={pipe_halluc:.2f})"
            )
        except Exception as e:
            logger.warning(f"  Condition D failed: {e}")
            rec["D"] = {"hallucination_rate": 1.0, "mean_entailment": 0.0,
                        "latency_s": 0.0, "n_claims": 0, "claim_scores": [],
                        "answer": "", "predicted_decision": "unknown", "correct": False,
                        "docs_retrieved": 0, "pipeline_hallucination_rate": 0.0}

        records.append(rec)

        # Save checkpoint every 10 questions
        if (i + 1) % 10 == 0:
            _save_checkpoint(records, output_path)
            logger.info(f"  Checkpoint saved ({i+1}/{total})")

    # ── Aggregate statistics ───────────────────────────────────────────────────
    logger.info("Computing aggregate statistics...")
    condition_stats = {}
    for cond in ["A", "B", "C", "D"]:
        h_rates = [r[cond]["hallucination_rate"] for r in records if cond in r]
        ent_scores = [r[cond]["mean_entailment"] for r in records if cond in r]
        latencies = [r[cond]["latency_s"] for r in records if cond in r]
        n_claims = [r[cond]["n_claims"] for r in records if cond in r]
        correct = [r[cond]["correct"] for r in records if cond in r]

        condition_stats[cond] = {
            "n": len(h_rates),
            "hallucination_rate_mean": round(float(np.mean(h_rates)), 4),
            "hallucination_rate_std": round(float(np.std(h_rates, ddof=1)) if len(h_rates) > 1 else 0.0, 4),
            "hallucination_rate_median": round(float(np.median(h_rates)), 4),
            "mean_entailment_mean": round(float(np.mean(ent_scores)), 4),
            "mean_entailment_std": round(float(np.std(ent_scores, ddof=1)) if len(ent_scores) > 1 else 0.0, 4),
            "avg_latency_s": round(float(np.mean(latencies)), 3),
            "p95_latency_s": round(float(np.percentile(latencies, 95)) if latencies else 0.0, 3),
            "avg_n_claims": round(float(np.mean(n_claims)), 2),
            "accuracy": round(float(np.mean(correct)), 4),
        }

    # ── Statistical comparisons ────────────────────────────────────────────────
    def get_rates(cond):
        return [r[cond]["hallucination_rate"] for r in records if cond in r]

    a_rates = get_rates("A")
    b_rates = get_rates("B")
    c_rates = get_rates("C")
    d_rates = get_rates("D")

    a_mean = float(np.mean(a_rates))
    d_mean = float(np.mean(d_rates))
    relative_reduction_ad = (a_mean - d_mean) / a_mean * 100 if a_mean > 0 else 0.0

    t_ad, p_ad = welch_t_test(a_rates, d_rates)
    t_cd, p_cd = welch_t_test(c_rates, d_rates)
    t_ac, p_ac = welch_t_test(a_rates, c_rates)
    t_bc, p_bc = welch_t_test(b_rates, c_rates)

    d_ad = cohen_d(a_rates, d_rates)
    d_cd = cohen_d(c_rates, d_rates)

    comparisons = {
        "A_vs_D_relative_reduction_pct": round(relative_reduction_ad, 2),
        "A_vs_D_t_statistic": t_ad,
        "A_vs_D_p_value": p_ad,
        "A_vs_D_significant_p05": p_ad < 0.05,
        "A_vs_D_cohens_d": d_ad,
        "A_vs_D_meets_50pct_target": relative_reduction_ad >= 50.0,
        "C_vs_D_t_statistic": t_cd,
        "C_vs_D_p_value": p_cd,
        "C_vs_D_significant_p05": p_cd < 0.05,
        "C_vs_D_cohens_d": d_cd,
        "A_vs_C_t_statistic": t_ac,
        "A_vs_C_p_value": p_ac,
        "A_vs_C_significant_p05": p_ac < 0.05,
        "B_vs_C_t_statistic": t_bc,
        "B_vs_C_p_value": p_bc,
        "B_vs_C_significant_p05": p_bc < 0.05,
    }

    summary = {
        "n_questions": len(records),
        "nli_model": settings.nli_model,
        "llm_model": settings.llm_model_name,
        "measurement_threshold": MEASUREMENT_THRESHOLD,
        "condition_stats": condition_stats,
        "comparisons": comparisons,
        "records": records,
    }

    # ── Save results ───────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or RESULTS_DIR / "four_condition_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    logger.success(f"Results saved → {out}")

    # Save per-condition JSONL for hallucination_eval.py compatibility
    for cond, cond_name in [
        ("A", "base_llm"),
        ("B", "rag_only"),
        ("C", "rag_reranker"),
        ("D", "rag_with_hallucination_filter"),
    ]:
        pred_path = Path("data/processed") / f"medqa_predictions_{cond_name}.jsonl"
        with open(pred_path, "w") as f:
            for r in records:
                if cond in r:
                    f.write(json.dumps({
                        "question": r["question"],
                        "gold_decision": r["gold_decision"],
                        "predicted_decision": r[cond].get("predicted_decision", "unknown"),
                        "correct": r[cond].get("correct", False),
                        "hallucination_rate": r[cond]["hallucination_rate"],
                        "mean_entailment": r[cond]["mean_entailment"],
                        "latency_s": r[cond]["latency_s"],
                        "answer": r[cond].get("answer", ""),
                    }) + "\n")
        logger.info(f"Saved predictions for {cond_name} → {pred_path}")

    # ── Log to MLflow ──────────────────────────────────────────────────────────
    log_to_mlflow(summary)

    # ── Print comparison table ─────────────────────────────────────────────────
    _print_table(summary)

    return summary


def _save_checkpoint(records: list, output_path: Path | None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = output_path or RESULTS_DIR / "four_condition_eval_checkpoint.json"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    with open(ckpt.with_suffix(".checkpoint.json"), "w") as f:
        json.dump({"records": records}, f)


def _print_table(summary: dict) -> None:
    """Print the comparison table as shown in Table 4.1 of the dissertation."""
    stats = summary["condition_stats"]
    cmp = summary["comparisons"]
    n = summary["n_questions"]

    cond_labels = {
        "A": "Cond A: Base LLM",
        "B": "Cond B: RAG (embed)",
        "C": "Cond C: RAG+Rerank",
        "D": "Cond D: Full System",
    }

    print()
    print("=" * 80)
    print(f"  TABLE 4.1 — Four-Condition Hallucination Evaluation  (n={n})")
    print(f"  NLI model: {summary['nli_model']}")
    print(f"  LLM: {summary['llm_model']} | Measurement threshold: {summary['measurement_threshold']}")
    print("=" * 80)
    print(
        f"  {'Condition':<22}  {'Halluc Rate':>12}  {'±SD':>8}  "
        f"{'Entailment':>11}  {'Accuracy':>9}  {'P95 Lat':>8}"
    )
    print(f"  {'-'*78}")
    for cond in ["A", "B", "C", "D"]:
        s = stats[cond]
        label = cond_labels[cond]
        print(
            f"  {label:<22}  {s['hallucination_rate_mean']:>12.4f}  "
            f"{s['hallucination_rate_std']:>8.4f}  "
            f"{s['mean_entailment_mean']:>11.4f}  "
            f"{s['accuracy']:>9.4f}  "
            f"{s['p95_latency_s']:>8.1f}s"
        )
    print(f"  {'-'*78}")
    print()

    # Primary hypothesis: A vs D
    red = cmp["A_vs_D_relative_reduction_pct"]
    sig = "significant" if cmp["A_vs_D_significant_p05"] else "not significant"
    target = "MEETS" if cmp["A_vs_D_meets_50pct_target"] else "BELOW"
    print(
        f"  H1-1 (≥50% relative halluc. reduction, A→D): "
        f"{red:.1f}%  [{target} target]"
    )
    print(
        f"       Welch t={cmp['A_vs_D_t_statistic']:.3f}, "
        f"p={cmp['A_vs_D_p_value']:.4f} ({sig}), "
        f"Cohen's d={cmp['A_vs_D_cohens_d']:.3f}"
    )
    print()

    # H1-3: Reranking ablation (C vs D — effect of NLI filter)
    sig_cd = "significant" if cmp["C_vs_D_significant_p05"] else "not significant"
    c_mean = stats["C"]["hallucination_rate_mean"]
    d_mean = stats["D"]["hallucination_rate_mean"]
    red_cd = (c_mean - d_mean) / c_mean * 100 if c_mean > 0 else 0.0
    print(
        f"  H1-3 (NLI filter effect, C→D):  "
        f"{red_cd:.1f}% reduction  "
        f"Welch t={cmp['C_vs_D_t_statistic']:.3f}, "
        f"p={cmp['C_vs_D_p_value']:.4f} ({sig_cd})"
    )

    # Reranker effect (B vs C)
    sig_bc = "significant" if cmp["B_vs_C_significant_p05"] else "not significant"
    print(
        f"  Reranking effect (B→C):          "
        f"Welch t={cmp['B_vs_C_t_statistic']:.3f}, "
        f"p={cmp['B_vs_C_p_value']:.4f} ({sig_bc})"
    )
    print()
    print("=" * 80)


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Four-condition hallucination evaluation")
    parser.add_argument("--n", type=int, default=50, help="Number of PubMedQA questions")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    settings.ensure_dirs()
    run_evaluation(n=args.n, output_path=args.output)


if __name__ == "__main__":
    main()
