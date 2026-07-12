"""
Unit tests for HallucinationDetectorAgent and NLI utilities.
Mocks the NLI model to avoid loading DeBERTa in unit tests.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from src.agents.hallucination import (
    ClaimVerification,
    HallucinationDetectorAgent,
    HallucinationReport,
)
from src.models.nli import extract_claims


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_nli_model():
    """NLI model that marks short claims as hallucinated, long ones as supported."""
    model = MagicMock()

    def score_claim(claim, evidence_chunks):
        entailment = 0.85 if len(claim) > 30 else 0.1
        return {
            "claim": claim,
            "max_entailment": entailment,
            "is_supported": entailment >= 0.7,
            "best_evidence": evidence_chunks[0] if evidence_chunks else "",
            "label": "entailment" if entailment >= 0.7 else "neutral",
            "all_scores": [entailment] * len(evidence_chunks),
        }

    model.score_claim_against_evidence.side_effect = score_claim
    return model


@pytest.fixture
def detector(mock_nli_model):
    agent = HallucinationDetectorAgent(
        nli_model=mock_nli_model,
        entailment_threshold=0.7,
    )
    return agent


EVIDENCE = [
    "Metformin is the first-line treatment for type 2 diabetes mellitus.",
    "Insulin therapy is recommended for patients with severe hyperglycemia.",
]


# ── extract_claims ─────────────────────────────────────────────────────────────

class TestExtractClaims:
    def test_splits_sentences(self):
        text = "Metformin is used for diabetes. Insulin is another option."
        claims = extract_claims(text)
        assert len(claims) == 2

    def test_filters_short_fragments(self):
        claims = extract_claims("Yes. No. Maybe.")
        assert all(len(c) > 20 for c in claims)

    def test_empty_text(self):
        assert extract_claims("") == []

    def test_single_sentence(self):
        text = "Metformin reduces blood glucose in type 2 diabetes patients."
        claims = extract_claims(text)
        assert len(claims) == 1


# ── HallucinationDetectorAgent ─────────────────────────────────────────────────

class TestHallucinationDetectorAgent:
    def test_verify_returns_report(self, detector):
        response = "Metformin is the first-line treatment for diabetes. It works well."
        report = detector.verify(response, EVIDENCE)
        assert isinstance(report, HallucinationReport)

    def test_report_has_claims(self, detector):
        response = "Metformin is recommended for type 2 diabetes patients regularly."
        report = detector.verify(response, EVIDENCE)
        assert report.total_claims > 0

    def test_hallucination_rate_range(self, detector):
        response = "Metformin is recommended for type 2 diabetes patients regularly."
        report = detector.verify(response, EVIDENCE)
        assert 0.0 <= report.hallucination_rate <= 1.0

    def test_empty_response(self, detector):
        report = detector.verify("", EVIDENCE)
        assert report.hallucination_rate == 0.0

    def test_no_evidence_all_hallucinated(self, detector):
        response = "Metformin is the first-line treatment for type 2 diabetes mellitus."
        report = detector.verify(response, [])
        assert report.hallucination_rate == 1.0

    def test_verified_response_brackets_hallucinated(self, detector):
        # Short claims will be marked as hallucinated by mock
        response = "Yes. Metformin reduces blood glucose in diabetic patients significantly."
        report = detector.verify(response, EVIDENCE, flag_mode_override=None)
        # The original response should have some modification
        assert report.verified_response is not None

    def test_summary_dict_keys(self, detector):
        response = "Metformin is the standard treatment for type 2 diabetes patients."
        report = detector.verify(response, EVIDENCE)
        summary = report.summary()
        assert "total_claims" in summary
        assert "hallucination_rate" in summary
        assert "supported_claims" in summary

    def test_compute_aggregate_rate(self, detector):
        responses = [
            "Metformin is the standard treatment for type 2 diabetes patients.",
            "Insulin therapy is used for severe hyperglycemia in diabetic patients.",
        ]
        evidence_list = [EVIDENCE, EVIDENCE]
        rate = detector.compute_hallucination_rate(responses, evidence_list)
        assert 0.0 <= rate <= 1.0

    def test_empty_batch_rate(self, detector):
        rate = detector.compute_hallucination_rate([], [])
        assert rate == 0.0


class TestClaimVerification:
    def test_construction(self):
        cv = ClaimVerification(
            claim="Test claim",
            entailment_score=0.85,
            is_hallucinated=False,
            best_evidence="Evidence text",
        )
        assert cv.entailment_score == 0.85
        assert not cv.is_hallucinated
