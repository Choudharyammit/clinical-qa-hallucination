"""
Unit tests for A/B test harness.

All orchestrator calls are mocked — no model loading required.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.ab_test import (
    ABTestRecord,
    ABTestResults,
    _welch_t_test,
    _t_cdf_approx,
    run_ab_test,
    load_ab_test_results,
)
from src.evaluation.hallucination_eval import compute_hallucination_stats


# ── Sample data ────────────────────────────────────────────────────────────────

SAMPLE_QUESTIONS = [
    {
        "question_id": f"q{i}",
        "question": f"What is treatment {i} for diabetes?",
        "answer": "A",
        "options": {"A": "Metformin", "B": "Insulin", "C": "Surgery", "D": "Aspirin"},
    }
    for i in range(10)
]


def make_mock_orchestrator(
    *,
    a_halluc: float = 0.4,
    b_halluc: float = 0.1,
    prediction: str = "A",
):
    """Return an orchestrator mock that returns configurable results."""
    orch = MagicMock()

    def run(question, mcq_options=None, use_rag=True, use_hallucination_filter=False, **kwargs):
        rate = b_halluc if use_hallucination_filter else a_halluc
        return {
            "answer": f"Answer to: {question}",
            "mcq_prediction": prediction,
            "hallucination_rate": rate,
            "confidence": "High",
            "response_time_s": 0.5,
            "citations": [],
            "sources_retrieved": 3,
            "iterations": 2,
        }

    orch.run.side_effect = run
    return orch


# ── _welch_t_test ──────────────────────────────────────────────────────────────

class TestWelchTTest:
    def test_different_means_gives_nonzero_t(self):
        a = [0.4] * 30 + [0.5] * 20
        b = [0.1] * 30 + [0.15] * 20
        t, p = _welch_t_test(a, b)
        assert abs(t) > 1.0
        assert 0.0 <= p <= 1.0

    def test_identical_lists_give_zero_t(self):
        a = [0.3] * 20
        b = [0.3] * 20
        t, p = _welch_t_test(a, b)
        assert t == 0.0
        assert p == 1.0

    def test_small_samples_returns_valid(self):
        t, p = _welch_t_test([0.4, 0.5], [0.1, 0.2])
        assert isinstance(t, float)
        assert 0.0 <= p <= 1.0

    def test_single_element_safe(self):
        t, p = _welch_t_test([0.5], [0.1])
        assert t == 0.0
        assert p == 1.0

    def test_empty_safe(self):
        t, p = _welch_t_test([], [])
        assert t == 0.0
        assert p == 1.0


class TestTCDFApprox:
    def test_large_t_small_p(self):
        p = _t_cdf_approx(10.0, 100.0)
        assert p < 0.001

    def test_zero_t_p_around_one(self):
        p = _t_cdf_approx(0.0, 30.0)
        assert p == pytest.approx(1.0, abs=0.01)

    def test_output_in_zero_one(self):
        for t in [0.0, 1.0, 2.0, 5.0]:
            for df in [5.0, 30.0, 100.0]:
                p = _t_cdf_approx(t, df)
                assert 0.0 <= p <= 1.0, f"p={p} out of range for t={t}, df={df}"


# ── ABTestRecord ───────────────────────────────────────────────────────────────

class TestABTestRecord:
    def test_to_dict_has_all_fields(self):
        rec = ABTestRecord(
            question_id="q1",
            question="What?",
            correct_answer="A",
            mcq_options={"A": "Metformin"},
        )
        d = rec.to_dict()
        for field in [
            "question_id", "question", "correct_answer",
            "a_hallucination_rate", "b_hallucination_rate",
            "a_correct", "b_correct",
        ]:
            assert field in d


# ── ABTestResults ──────────────────────────────────────────────────────────────

class TestABTestResults:
    def test_meets_50pct_target_true(self):
        r = ABTestResults(hallucination_reduction_pct=55.0)
        assert r.meets_50pct_target() is True

    def test_meets_50pct_target_false(self):
        r = ABTestResults(hallucination_reduction_pct=45.0)
        assert r.meets_50pct_target() is False

    def test_to_dict_excludes_records(self):
        r = ABTestResults(n=5)
        d = r.to_dict()
        assert "records" not in d
        assert "n" in d

    def test_print_summary_no_crash(self, capsys):
        r = ABTestResults(n=10, a_accuracy=0.6, b_accuracy=0.65,
                          a_hallucination_mean=0.4, b_hallucination_mean=0.15,
                          hallucination_reduction_pct=62.5,
                          t_statistic=3.2, p_value=0.002, significant=True)
        r.print_summary()
        captured = capsys.readouterr()
        assert "62.5" in captured.out


# ── run_ab_test ────────────────────────────────────────────────────────────────

class TestRunABTest:
    def test_returns_ab_test_results(self, tmp_path):
        orch = make_mock_orchestrator(a_halluc=0.4, b_halluc=0.1)
        results = run_ab_test(
            SAMPLE_QUESTIONS, orch, n=5,
            output_dir=tmp_path, log_mlflow=False
        )
        assert isinstance(results, ABTestResults)

    def test_n_records_correct(self, tmp_path):
        orch = make_mock_orchestrator()
        results = run_ab_test(
            SAMPLE_QUESTIONS, orch, n=5,
            output_dir=tmp_path, log_mlflow=False
        )
        assert results.n == 5

    def test_hallucination_reduction_computed(self, tmp_path):
        orch = make_mock_orchestrator(a_halluc=0.4, b_halluc=0.1)
        results = run_ab_test(
            SAMPLE_QUESTIONS, orch, n=5,
            output_dir=tmp_path, log_mlflow=False
        )
        # reduction should be ~75% (0.4 → 0.1)
        assert results.hallucination_reduction_pct == pytest.approx(75.0, abs=2.0)

    def test_meets_target_when_large_reduction(self, tmp_path):
        orch = make_mock_orchestrator(a_halluc=0.5, b_halluc=0.1)
        results = run_ab_test(
            SAMPLE_QUESTIONS, orch, n=5,
            output_dir=tmp_path, log_mlflow=False
        )
        assert results.meets_50pct_target()

    def test_accuracy_computed(self, tmp_path):
        orch = make_mock_orchestrator(prediction="A")  # all correct (answer is A)
        results = run_ab_test(
            SAMPLE_QUESTIONS, orch, n=5,
            output_dir=tmp_path, log_mlflow=False
        )
        assert results.a_accuracy == pytest.approx(1.0, abs=0.01)
        assert results.b_accuracy == pytest.approx(1.0, abs=0.01)

    def test_saves_jsonl(self, tmp_path):
        orch = make_mock_orchestrator()
        run_ab_test(SAMPLE_QUESTIONS, orch, n=5, output_dir=tmp_path, log_mlflow=False)
        records_file = tmp_path / "ab_test_records.jsonl"
        assert records_file.exists()
        records = [json.loads(l) for l in records_file.read_text().splitlines() if l.strip()]
        assert len(records) == 5

    def test_saves_summary_json(self, tmp_path):
        orch = make_mock_orchestrator()
        run_ab_test(SAMPLE_QUESTIONS, orch, n=5, output_dir=tmp_path, log_mlflow=False)
        summary_file = tmp_path / "ab_test_summary.json"
        assert summary_file.exists()
        d = json.loads(summary_file.read_text())
        assert "hallucination_reduction_pct" in d

    def test_n_capped_at_available_questions(self, tmp_path):
        orch = make_mock_orchestrator()
        results = run_ab_test(
            SAMPLE_QUESTIONS[:3], orch, n=100,
            output_dir=tmp_path, log_mlflow=False
        )
        assert results.n == 3

    def test_orchestrator_error_recovery(self, tmp_path):
        """Error on condition A should not crash — rec gets zero rate."""
        orch = MagicMock()
        call_count = [0]

        def run_side_effect(q, **kwargs):
            call_count[0] += 1
            if not kwargs.get("use_hallucination_filter"):
                raise RuntimeError("Simulated OOM")
            return {"answer": "ok", "mcq_prediction": "A", "hallucination_rate": 0.1,
                    "confidence": "High", "response_time_s": 0.5, "citations": [],
                    "sources_retrieved": 2, "iterations": 1}

        orch.run.side_effect = run_side_effect
        results = run_ab_test(
            SAMPLE_QUESTIONS[:3], orch, n=3,
            output_dir=tmp_path, log_mlflow=False
        )
        assert results.n == 3  # did not crash


# ── load_ab_test_results ───────────────────────────────────────────────────────

class TestLoadABTestResults:
    def test_returns_none_when_no_file(self, tmp_path):
        result = load_ab_test_results(tmp_path / "nonexistent.json")
        assert result is None

    def test_loads_saved_summary(self, tmp_path):
        orch = make_mock_orchestrator(a_halluc=0.4, b_halluc=0.15)
        results = run_ab_test(
            SAMPLE_QUESTIONS[:4], orch, n=4,
            output_dir=tmp_path, log_mlflow=False
        )
        summary_path = tmp_path / "ab_test_summary.json"
        loaded = load_ab_test_results(summary_path)
        assert loaded is not None
        assert loaded.n == results.n
        assert loaded.hallucination_reduction_pct == pytest.approx(
            results.hallucination_reduction_pct, abs=0.1
        )


# ── compute_hallucination_stats ────────────────────────────────────────────────

class TestComputeHallucinationStats:
    def test_empty_returns_empty_dict(self):
        assert compute_hallucination_stats([]) == {}

    def test_basic_stats_keys(self):
        preds = [{"hallucination_rate": r} for r in [0.0, 0.2, 0.4, 0.6, 0.8]]
        stats = compute_hallucination_stats(preds)
        for key in ["mean", "median", "std", "p25", "p75", "p95", "n", "pct_zero_halluc"]:
            assert key in stats

    def test_mean_correct(self):
        preds = [{"hallucination_rate": r} for r in [0.1, 0.3, 0.5]]
        stats = compute_hallucination_stats(preds)
        assert stats["mean"] == pytest.approx(0.3, abs=0.001)

    def test_pct_zero_halluc(self):
        preds = [{"hallucination_rate": 0.0}] * 3 + [{"hallucination_rate": 0.5}] * 7
        stats = compute_hallucination_stats(preds)
        assert stats["pct_zero_halluc"] == pytest.approx(0.3, abs=0.001)

    def test_missing_halluc_defaults_zero(self):
        preds = [{"mcq_prediction": "A"}] * 5
        stats = compute_hallucination_stats(preds)
        assert stats["mean"] == 0.0
