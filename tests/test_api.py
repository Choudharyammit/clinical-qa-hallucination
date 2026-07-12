"""
Integration tests for the FastAPI REST API.
Uses TestClient with mocked orchestrator to test endpoints without loading models.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


# ── Mock orchestrator output ───────────────────────────────────────────────────

MOCK_RESPONSE = {
    "answer": "Metformin is the first-line treatment for type 2 diabetes.",
    "mcq_prediction": "A",
    "confidence": "High",
    "hallucination_rate": 0.05,
    "response_time_s": 1.23,
    "citations": [
        {
            "source": "PubMed:12345",
            "title": "Diabetes Management Guidelines",
            "pmid": "12345",
            "year": "2023",
            "relevance": 0.92,
        }
    ],
    "hallucination_report": {
        "total_claims": 3,
        "supported_claims": 2,
        "hallucinated_claims": 1,
        "hallucination_rate": 0.05,
    },
    "sources_retrieved": 5,
    "iterations": 2,
}


@pytest.fixture
def client():
    """Test client with mocked orchestrator."""
    mock_orch = MagicMock()
    mock_orch.run.return_value = MOCK_RESPONSE

    with TestClient(app) as c:
        app.state.orchestrator = mock_orch
        yield c


# ── /health ────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_schema(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_health_model_loaded_true(self, client):
        resp = client.get("/health")
        assert resp.json()["model_loaded"] is True


# ── /query ─────────────────────────────────────────────────────────────────────

class TestQueryEndpoint:
    def test_basic_query_returns_200(self, client):
        resp = client.post("/query", json={"question": "What is the treatment for diabetes?"})
        assert resp.status_code == 200

    def test_response_has_answer(self, client):
        resp = client.post("/query", json={"question": "What is the treatment for diabetes?"})
        data = resp.json()
        assert "answer" in data
        assert len(data["answer"]) > 0

    def test_response_has_citations(self, client):
        resp = client.post("/query", json={"question": "Treatment for hypertension?"})
        data = resp.json()
        assert "citations" in data
        assert isinstance(data["citations"], list)

    def test_response_has_hallucination_rate(self, client):
        resp = client.post("/query", json={"question": "Treatment for diabetes?"})
        data = resp.json()
        assert "hallucination_rate" in data
        assert 0.0 <= data["hallucination_rate"] <= 1.0

    def test_response_has_response_time(self, client):
        resp = client.post("/query", json={"question": "How is pneumonia treated?"})
        data = resp.json()
        assert "response_time_s" in data

    def test_mcq_query(self, client):
        resp = client.post("/query", json={
            "question": "First-line treatment for type 2 diabetes?",
            "mcq_options": {"A": "Metformin", "B": "Insulin", "C": "Surgery", "D": "Aspirin"},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "mcq_prediction" in data

    def test_query_too_short_returns_422(self, client):
        resp = client.post("/query", json={"question": "Hi"})
        assert resp.status_code == 422

    def test_empty_question_returns_422(self, client):
        resp = client.post("/query", json={"question": ""})
        assert resp.status_code == 422

    def test_rag_disabled(self, client):
        resp = client.post("/query", json={
            "question": "What is the first-line treatment for hypertension?",
            "use_rag": False,
        })
        assert resp.status_code == 200

    def test_process_time_header(self, client):
        resp = client.post("/query", json={"question": "What is diabetes mellitus type 2?"})
        assert "X-Process-Time" in resp.headers


# ── /metrics ───────────────────────────────────────────────────────────────────

class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_schema(self, client):
        resp = client.get("/metrics")
        data = resp.json()
        assert "experiment_name" in data


# ── Error handling ─────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_orchestrator_error_returns_500(self, client):
        app.state.orchestrator.run.side_effect = RuntimeError("Model OOM")
        resp = client.post("/query", json={"question": "What is the treatment for diabetes?"})
        assert resp.status_code == 500
        # Reset for other tests
        app.state.orchestrator.run.side_effect = None
        app.state.orchestrator.run.return_value = MOCK_RESPONSE

    def test_no_orchestrator_returns_503(self):
        with TestClient(app) as c:
            app.state.orchestrator = None
            resp = c.post("/query", json={"question": "What is the treatment for diabetes?"})
            assert resp.status_code == 503
