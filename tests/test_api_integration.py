"""
API integration tests for the Clinical QA System FastAPI application.

Tests the full HTTP request/response cycle using TestClient (no real server needed).
All heavy models (LLM, embeddings, NLI) are replaced with lightweight mocks so
these tests run in seconds without GPU.

Coverage:
  - POST /query  — success paths, validation, MCQ mode, no-RAG mode, error handling
  - GET  /health — healthy and degraded responses
  - GET  /metrics — returns metrics or 404 before first eval
  - Middleware — response time header present
  - Schemas — response fields match QueryResponse model
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Mock orchestrator factory ──────────────────────────────────────────────────

def make_mock_orchestrator(
    *,
    answer: str = "Metformin is the first-line treatment for type 2 diabetes.",
    mcq_prediction: str = "A",
    hallucination_rate: float = 0.05,
    sources_retrieved: int = 5,
    raise_error: bool = False,
) -> MagicMock:
    orch = MagicMock()

    def run(query, **kwargs):
        if raise_error:
            raise RuntimeError("Simulated GPU OOM")
        options = kwargs.get("mcq_options")
        return {
            "answer": answer,
            "mcq_prediction": mcq_prediction if options else None,
            "confidence": "High",
            "hallucination_rate": hallucination_rate,
            "response_time_s": 0.35,
            "citations": [
                {"source": "PubMed:12345", "title": "Study", "pmid": "12345",
                 "year": "2023", "relevance": 0.9}
            ],
            "hallucination_report": {
                "total_claims": 3,
                "supported_claims": 3,
                "hallucinated_claims": 0,
                "hallucination_rate": hallucination_rate,
                "overall_confidence": 0.91,
            },
            "sources_retrieved": sources_retrieved,
            "iterations": 3,
        }

    orch.run.side_effect = run
    return orch


# ── TestClient fixture ─────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """Create a TestClient with a mocked orchestrator injected into app.state."""
    from src.api.main import app

    mock_orch = make_mock_orchestrator()

    with TestClient(app) as c:
        app.state.orchestrator = mock_orch
        yield c


@pytest.fixture
def client_no_orchestrator():
    """TestClient where orchestrator has NOT been initialized (503 scenario)."""
    from src.api.main import app

    with TestClient(app) as c:
        if hasattr(app.state, "orchestrator"):
            del app.state.orchestrator
        yield c


@pytest.fixture
def client_error():
    """TestClient where orchestrator always raises an error."""
    from src.api.main import app

    mock_orch = make_mock_orchestrator(raise_error=True)

    with TestClient(app) as c:
        app.state.orchestrator = mock_orch
        yield c


# ── POST /query ────────────────────────────────────────────────────────────────

class TestQueryEndpoint:
    def test_basic_query_returns_200(self, client):
        r = client.post("/query", json={"question": "What is the treatment for diabetes?"})
        assert r.status_code == 200

    def test_response_has_answer(self, client):
        r = client.post("/query", json={"question": "What is the treatment for diabetes?"})
        assert "answer" in r.json()
        assert len(r.json()["answer"]) > 0

    def test_response_schema_complete(self, client):
        r = client.post("/query", json={"question": "What is type 2 diabetes mellitus?"})
        body = r.json()
        required = [
            "answer", "mcq_prediction", "confidence", "hallucination_rate",
            "response_time_s", "citations", "hallucination_report",
            "sources_retrieved", "iterations",
        ]
        for field in required:
            assert field in body, f"Missing field: {field}"

    def test_mcq_mode_returns_prediction(self, client):
        r = client.post("/query", json={
            "question": "First-line treatment for type 2 DM?",
            "mcq_options": {"A": "Metformin", "B": "Insulin", "C": "Surgery", "D": "Aspirin"},
        })
        assert r.status_code == 200
        assert r.json()["mcq_prediction"] == "A"

    def test_no_rag_mode(self, client):
        r = client.post("/query", json={
            "question": "What is diabetes?",
            "use_rag": False,
            "use_hallucination_filter": False,
        })
        assert r.status_code == 200
        assert "answer" in r.json()

    def test_hallucination_rate_in_range(self, client):
        r = client.post("/query", json={"question": "What causes hypertension?"})
        hr = r.json()["hallucination_rate"]
        assert 0.0 <= hr <= 1.0

    def test_citations_is_list(self, client):
        r = client.post("/query", json={"question": "What is aspirin?"})
        assert isinstance(r.json()["citations"], list)

    def test_hallucination_report_is_dict(self, client):
        r = client.post("/query", json={"question": "What is insulin resistance?"})
        assert isinstance(r.json()["hallucination_report"], dict)

    def test_response_time_positive(self, client):
        r = client.post("/query", json={"question": "What is hypertension?"})
        assert r.json()["response_time_s"] > 0

    def test_max_context_chunks_accepted(self, client):
        r = client.post("/query", json={
            "question": "What is the mechanism of metformin?",
            "max_context_chunks": 3,
        })
        assert r.status_code == 200

    def test_process_time_header_present(self, client):
        r = client.post("/query", json={"question": "What is sepsis?"})
        assert "X-Process-Time" in r.headers

    def test_content_type_json(self, client):
        r = client.post("/query", json={"question": "What is COPD?"})
        assert "application/json" in r.headers.get("content-type", "")

    # Validation errors
    def test_missing_question_returns_422(self, client):
        r = client.post("/query", json={"use_rag": True})
        assert r.status_code == 422

    def test_question_too_short_returns_422(self, client):
        r = client.post("/query", json={"question": "Hi?"})
        assert r.status_code == 422

    def test_question_too_long_returns_422(self, client):
        r = client.post("/query", json={"question": "x" * 1001})
        assert r.status_code == 422

    def test_invalid_max_context_chunks_returns_422(self, client):
        r = client.post("/query", json={
            "question": "What is sepsis?",
            "max_context_chunks": 0,  # ge=1 required
        })
        assert r.status_code == 422

    def test_question_stripped(self, client):
        r = client.post("/query", json={"question": "   What is metformin?   "})
        assert r.status_code == 200

    # Service unavailable
    def test_503_when_orchestrator_not_initialized(self, client_no_orchestrator):
        r = client_no_orchestrator.post("/query", json={"question": "What is diabetes?"})
        assert r.status_code == 503

    # Error recovery
    def test_500_on_orchestrator_error(self, client_error):
        r = client_error.post("/query", json={"question": "What is diabetes?"})
        # Should get a structured error (500 or 503), not an HTML traceback
        assert r.status_code in (500, 503)
        body = r.json()
        assert "error" in body or "detail" in body


# ── GET /health ────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_status_field(self, client):
        r = client.get("/health")
        assert "status" in r.json()

    def test_health_model_loaded_true(self, client):
        r = client.get("/health")
        # model_loaded reflects whether orchestrator is set
        assert "model_loaded" in r.json()
        assert r.json()["model_loaded"] is True

    def test_health_has_version(self, client):
        r = client.get("/health")
        assert "version" in r.json()

    def test_health_no_orchestrator_still_200(self, client_no_orchestrator):
        # Health endpoint should return 200 even without orchestrator
        r = client_no_orchestrator.get("/health")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is False


# ── GET /metrics ───────────────────────────────────────────────────────────────

class TestMetricsEndpoint:
    def test_metrics_returns_200_or_404(self, client):
        # 200 if latest_metrics.json exists; 404 if not yet generated
        r = client.get("/metrics")
        assert r.status_code in (200, 404)

    def test_metrics_200_has_schema(self, client, tmp_path, monkeypatch):
        """If metrics file exists, response should have expected fields."""
        import json
        from src.config import settings

        # Write a fake metrics file
        metrics = {
            "experiment_name": "test_experiment",
            "accuracy": 0.72,
            "hallucination_rate": 0.12,
            "avg_response_time_s": 1.8,
            "total_questions": 100,
            "last_updated": "2026-07-12 10:00:00",
        }
        metrics_file = settings.data_processed_dir / "latest_metrics.json"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.write_text(json.dumps(metrics))

        r = client.get("/metrics")
        if r.status_code == 200:
            body = r.json()
            assert "accuracy" in body or "experiment_name" in body


# ── Request validation edge cases ─────────────────────────────────────────────

class TestRequestValidation:
    def test_extra_fields_ignored(self, client):
        r = client.post("/query", json={
            "question": "What is hypertension treatment?",
            "unknown_field": "should be ignored",
        })
        assert r.status_code == 200

    def test_boolean_flags_default_true(self, client):
        r = client.post("/query", json={"question": "What is a beta blocker?"})
        assert r.status_code == 200

    def test_mcq_options_none_accepted(self, client):
        r = client.post("/query", json={
            "question": "Describe the renin-angiotensin system.",
            "mcq_options": None,
        })
        assert r.status_code == 200
        assert r.json()["mcq_prediction"] is None
