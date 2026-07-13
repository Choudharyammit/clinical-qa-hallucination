"""
Locust load test for the Clinical QA System FastAPI.

Tests the /query endpoint under concurrent user load to verify:
  - P95 response time < 3s (thesis SLA)
  - 0% error rate under 10 concurrent users
  - /health endpoint always returns 200

Usage:
    # Headless — results to JSON:
    locust -f locustfile.py --headless -u 10 -r 2 -t 60s \
        --host http://localhost:8000 --json --logfile load_test.log

    # Web UI:
    locust -f locustfile.py --host http://localhost:8000
    # then open http://localhost:8089

    # Single-user smoke test (no API server needed for unit tests):
    python scripts/run_load_test.py --smoke
"""

import json
import random
import time
from typing import Optional

from locust import HttpUser, TaskSet, between, events, task
from locust.env import Environment
from loguru import logger


# ── Sample clinical questions ──────────────────────────────────────────────────

CLINICAL_QUESTIONS = [
    {
        "question": "What is the first-line treatment for type 2 diabetes mellitus?",
        "mcq_options": {
            "A": "Metformin",
            "B": "Insulin glargine",
            "C": "Glibenclamide",
            "D": "Pioglitazone",
        },
    },
    {
        "question": "Which antibiotic is the drug of choice for community-acquired pneumonia in a healthy adult?",
        "mcq_options": {
            "A": "Amoxicillin",
            "B": "Vancomycin",
            "C": "Meropenem",
            "D": "Gentamicin",
        },
    },
    {
        "question": "What is the mechanism of action of ACE inhibitors in hypertension?",
        "mcq_options": {
            "A": "Block angiotensin II receptor",
            "B": "Inhibit conversion of angiotensin I to angiotensin II",
            "C": "Inhibit aldosterone secretion directly",
            "D": "Block beta-adrenergic receptors",
        },
    },
    {
        "question": "Which test is most specific for diagnosing pulmonary embolism?",
        "mcq_options": {
            "A": "D-dimer",
            "B": "CT pulmonary angiography",
            "C": "Chest X-ray",
            "D": "Troponin",
        },
    },
    {
        "question": "What is the pathophysiology of myocardial infarction?",
        "mcq_options": None,  # Free-form question (no MCQ)
    },
    {
        "question": "Which gene mutation is associated with hereditary breast cancer?",
        "mcq_options": {
            "A": "KRAS",
            "B": "BRCA1/BRCA2",
            "C": "TP53 only",
            "D": "RET",
        },
    },
    {
        "question": "What are the cardinal features of Parkinson's disease?",
        "mcq_options": None,
    },
    {
        "question": "Which class of drugs is preferred for heart failure with reduced ejection fraction?",
        "mcq_options": {
            "A": "Beta blockers only",
            "B": "ACE inhibitors + beta blockers + mineralocorticoid antagonists",
            "C": "Calcium channel blockers",
            "D": "Digoxin monotherapy",
        },
    },
]


# ── Task sets ──────────────────────────────────────────────────────────────────

class HealthCheckTaskSet(TaskSet):
    @task(1)
    def health_check(self):
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")


class QueryTaskSet(TaskSet):
    """Task set for the main /query endpoint with different configurations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._question_idx = 0

    def _next_question(self) -> dict:
        q = CLINICAL_QUESTIONS[self._question_idx % len(CLINICAL_QUESTIONS)]
        self._question_idx += 1
        return q

    @task(5)
    def query_rag_mcq(self):
        """RAG + hallucination filter, MCQ mode (most realistic condition)."""
        q = self._next_question()
        payload = {
            "question": q["question"],
            "mcq_options": q["mcq_options"] or {"A": "True", "B": "False", "C": "Unknown", "D": "Other"},
            "use_rag": True,
            "use_hallucination_filter": True,
            "max_context_chunks": 5,
        }
        with self.client.post("/query", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "answer" not in data:
                    response.failure("Response missing 'answer' field")
                else:
                    response.success()
            elif response.status_code == 503:
                response.failure("Service unavailable — models not loaded")
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(3)
    def query_rag_no_filter(self):
        """RAG without hallucination filter (faster baseline)."""
        q = self._next_question()
        payload = {
            "question": q["question"],
            "use_rag": True,
            "use_hallucination_filter": False,
            "max_context_chunks": 3,
        }
        with self.client.post("/query", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @task(2)
    def query_base_llm(self):
        """Base LLM mode — no retrieval."""
        q = self._next_question()
        payload = {
            "question": q["question"],
            "use_rag": False,
            "use_hallucination_filter": False,
        }
        with self.client.post("/query", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @task(1)
    def health_check(self):
        """Intersperse health checks to detect service degradation."""
        self.client.get("/health")


class ClinicalQAUser(HttpUser):
    """Main user class — simulates a clinical professional using the QA system."""

    tasks = [QueryTaskSet]
    wait_time = between(1, 3)  # Think time: 1–3s between requests

    def on_start(self):
        """Verify the service is ready before starting load."""
        with self.client.get("/health", catch_response=True) as r:
            if r.status_code != 200:
                logger.warning("Service not ready — /health returned non-200")


class HealthOnlyUser(HttpUser):
    """Lightweight user for health-check-only load testing."""

    tasks = [HealthCheckTaskSet]
    wait_time = between(0.5, 1.0)


# ── Locust event hooks ─────────────────────────────────────────────────────────

@events.request.add_listener
def on_request(
    request_type: str,
    name: str,
    response_time: float,
    response_length: int,
    response,
    context,
    exception: Optional[Exception],
    **kwargs,
) -> None:
    """Flag any request exceeding the 3s SLA."""
    if name == "/query" and response_time > 3000:  # ms
        logger.warning(f"SLA BREACH: /query took {response_time:.0f}ms (> 3000ms SLA)")


@events.test_stop.add_listener
def on_test_stop(environment: Environment, **kwargs) -> None:
    """Print SLA summary when test completes."""
    stats = environment.runner.stats

    print(f"\n{'═'*65}")
    print("  Load Test SLA Report")
    print(f"{'═'*65}")

    for name, entry in stats.entries.items():
        p95 = entry.get_response_time_percentile(0.95)
        err_rate = entry.fail_ratio * 100
        sla_ok = p95 < 3000 if p95 else True
        print(
            f"  {name[1]:<10} "
            f"req={entry.num_requests:>6}  "
            f"p95={p95:>7.0f}ms  "
            f"err={err_rate:>5.1f}%  "
            f"{'✅ SLA OK' if sla_ok else '❌ SLA BREACH'}"
        )

    total = stats.total
    if total.num_requests > 0:
        overall_p95 = total.get_response_time_percentile(0.95)
        print(f"\n  Overall: {total.num_requests} requests, "
              f"p95={overall_p95:.0f}ms, "
              f"errors={total.fail_ratio*100:.1f}%")
        print(f"  {'✅ PASS' if overall_p95 < 3000 and total.fail_ratio == 0 else '❌ FAIL'}: "
              f"p95 < 3000ms AND 0% errors")

    print(f"{'═'*65}\n")
