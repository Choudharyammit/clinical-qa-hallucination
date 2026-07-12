"""
Unit tests for evaluation modules.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.evaluator import BenchmarkResult, EvaluatorAgent
from src.evaluation.hallucination_eval import compute_hallucination_stats


# ── BenchmarkResult ────────────────────────────────────────────────────────────

class TestBenchmarkResult:
    def test_to_dict_has_required_keys(self):
        result = BenchmarkResult(
            experiment_name="test",
            accuracy=0.75,
            hallucination_rate=0.10,
            avg_response_time_s=1.5,
            total_questions=100,
            correct_answers=75,
        )
        d = result.to_dict()
        assert d["accuracy"] == 0.75
        assert d["hallucination_rate"] == 0.10
        assert d["total_questions"] == 100


# ── compute_hallucination_stats ─────────────────────────────────────────────────

class TestComputeHallucinationStats:
    def test_basic_stats(self):
        preds = [
            {"hallucination_rate": 0.2},
            {"hallucination_rate": 0.4},
            {"hallucination_rate": 0.0},
        ]
        stats = compute_hallucination_stats(preds)
        assert abs(stats["mean"] - 0.2) < 1e-6
        assert stats["n"] == 3
        assert "p95" in stats

    def test_empty_predictions(self):
        stats = compute_hallucination_stats([])
        assert stats == {}

    def test_all_zero_hallucination(self):
        preds = [{"hallucination_rate": 0.0}] * 10
        stats = compute_hallucination_stats(preds)
        assert stats["mean"] == 0.0
        assert stats["pct_zero_halluc"] == 1.0


# ── EvaluatorAgent ─────────────────────────────────────────────────────────────

class TestEvaluatorAgent:
    def test_init_without_orchestrator(self):
        with patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"):
            agent = EvaluatorAgent()
            assert agent._orchestrator is None

    def test_set_orchestrator(self):
        with patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"):
            agent = EvaluatorAgent()
            mock_orch = MagicMock()
            agent.set_orchestrator(mock_orch)
            assert agent._orchestrator is mock_orch

    def test_load_medqa_test_missing_file(self, tmp_path):
        with patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"):
            agent = EvaluatorAgent()
            # Point to non-existent path
            with patch("src.agents.evaluator.settings") as mock_settings:
                mock_settings.data_processed_dir = tmp_path
                mock_settings.data_raw_dir = tmp_path
                result = agent.load_medqa_test()
                assert result == []

    def test_load_medqa_test_from_file(self, tmp_path):
        # Create a fake test file
        test_file = tmp_path / "medqa_test.jsonl"
        samples = [
            {"question": "Q1?", "options": {"A": "a", "B": "b"}, "answer_idx": "A"},
            {"question": "Q2?", "options": {"A": "c", "B": "d"}, "answer_idx": "B"},
        ]
        with open(test_file, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        with patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"):
            agent = EvaluatorAgent()
            with patch("src.agents.evaluator.settings") as mock_settings:
                mock_settings.data_processed_dir = tmp_path
                mock_settings.data_raw_dir = tmp_path
                result = agent.load_medqa_test(path=test_file, max_samples=1)
                assert len(result) == 1
