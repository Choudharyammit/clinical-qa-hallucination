"""
Unit tests for CitationAgent and Citation dataclass.
"""

import pytest

from src.agents.citation import Citation, CitationAgent, CitedResponse
from src.agents.hallucination import ClaimVerification, HallucinationReport
from src.agents.reranker import RankedDocument
from src.agents.retriever import RetrievedDocument


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def agent():
    return CitationAgent()


@pytest.fixture
def retrieved_docs():
    return [
        RetrievedDocument(
            text="Metformin is the first-line treatment for type 2 diabetes.",
            source="PubMed:11111",
            score=0.92,
            metadata={"pmid": "11111", "title": "Diabetes Management", "year": "2022", "journal": "NEJM"},
            rank=1,
        ),
        RetrievedDocument(
            text="ACE inhibitors reduce cardiovascular risk in diabetic patients.",
            source="PubMed:22222",
            score=0.78,
            metadata={"pmid": "22222", "title": "Cardiovascular Protection", "year": "2021", "journal": "Lancet"},
            rank=2,
        ),
    ]


@pytest.fixture
def ranked_docs():
    return [
        RankedDocument(
            text="Metformin reduces hepatic glucose production.",
            source="PubMed:33333",
            cross_encoder_score=0.88,
            original_rank=1,
            new_rank=1,
            metadata={"pmid": "33333", "title": "Metformin Mechanism", "year": "2020", "journal": "Diabetes Care"},
        ),
    ]


@pytest.fixture
def hallucination_report_with_supported(retrieved_docs):
    claims = [
        ClaimVerification(
            claim="Metformin is the first-line treatment for type 2 diabetes.",
            entailment_score=0.91,
            is_hallucinated=False,
            best_evidence="Metformin is the first-line treatment for type 2 diabetes.",
            evidence_source="PubMed:11111",
        ),
        ClaimVerification(
            claim="Insulin is rarely used today.",
            entailment_score=0.12,
            is_hallucinated=True,
            best_evidence="",
        ),
    ]
    return HallucinationReport(
        original_response="Metformin is the first-line treatment for type 2 diabetes. Insulin is rarely used today.",
        claims=claims,
        hallucination_rate=0.5,
        overall_confidence=0.51,
        verified_response="Metformin is the first-line treatment for type 2 diabetes. [UNVERIFIED: Insulin is rarely used today.]",
        total_claims=2,
        hallucinated_count=1,
        supported_count=1,
    )


# ── Citation dataclass ─────────────────────────────────────────────────────────

class TestCitation:
    def test_to_reference_string_full(self):
        cit = Citation(
            source="PubMed:12345",
            pmid="12345",
            title="Diabetes Management Guidelines",
            year="2023",
            journal="NEJM",
            relevance_score=0.9,
        )
        ref = cit.to_reference_string()
        assert "Diabetes Management Guidelines" in ref
        assert "NEJM" in ref
        assert "2023" in ref
        assert "12345" in ref

    def test_to_reference_string_partial(self):
        cit = Citation(source="PubMed:1", pmid="", title="", year="", journal="", relevance_score=0.5)
        ref = cit.to_reference_string()
        assert ref == "PubMed:1"


# ── CitationAgent.attach_citations ────────────────────────────────────────────

class TestAttachCitations:
    def test_returns_cited_response(self, agent, hallucination_report_with_supported, retrieved_docs):
        cited = agent.attach_citations(
            response=hallucination_report_with_supported.verified_response,
            hallucination_report=hallucination_report_with_supported,
            retrieved_docs=retrieved_docs,
        )
        assert isinstance(cited, CitedResponse)

    def test_citations_list_populated(self, agent, hallucination_report_with_supported, retrieved_docs):
        cited = agent.attach_citations(
            response=hallucination_report_with_supported.verified_response,
            hallucination_report=hallucination_report_with_supported,
            retrieved_docs=retrieved_docs,
        )
        assert len(cited.references) >= 1

    def test_inline_citation_marker_inserted(self, agent, hallucination_report_with_supported, retrieved_docs):
        cited = agent.attach_citations(
            response=hallucination_report_with_supported.verified_response,
            hallucination_report=hallucination_report_with_supported,
            retrieved_docs=retrieved_docs,
        )
        assert "[1]" in cited.text

    def test_full_response_has_references_section(self, agent, hallucination_report_with_supported, retrieved_docs):
        cited = agent.attach_citations(
            response=hallucination_report_with_supported.verified_response,
            hallucination_report=hallucination_report_with_supported,
            retrieved_docs=retrieved_docs,
        )
        assert "References:" in cited.full_response()

    def test_no_citation_for_hallucinated_claim(self, agent, hallucination_report_with_supported, retrieved_docs):
        # Only 1 supported claim → only 1 unique citation
        cited = agent.attach_citations(
            response=hallucination_report_with_supported.verified_response,
            hallucination_report=hallucination_report_with_supported,
            retrieved_docs=retrieved_docs,
        )
        assert len(cited.references) == 1  # Only the supported claim gets cited

    def test_ranked_docs_handled(self, agent, hallucination_report_with_supported, ranked_docs):
        cited = agent.attach_citations(
            response=hallucination_report_with_supported.verified_response,
            hallucination_report=hallucination_report_with_supported,
            retrieved_docs=ranked_docs,
        )
        assert isinstance(cited, CitedResponse)


# ── CitationAgent.attach_simple_citations ─────────────────────────────────────

class TestAttachSimpleCitations:
    def test_returns_cited_response(self, agent, retrieved_docs):
        cited = agent.attach_simple_citations("Test answer.", retrieved_docs)
        assert isinstance(cited, CitedResponse)

    def test_references_match_docs(self, agent, retrieved_docs):
        cited = agent.attach_simple_citations("Test.", retrieved_docs)
        assert len(cited.references) == len(retrieved_docs)

    def test_ranked_doc_score(self, agent, ranked_docs):
        cited = agent.attach_simple_citations("Test.", ranked_docs)
        # cross_encoder_score should be used, not 0.0
        assert cited.references[0].relevance_score == pytest.approx(0.88, abs=1e-3)

    def test_retrieved_doc_score(self, agent, retrieved_docs):
        cited = agent.attach_simple_citations("Test.", retrieved_docs)
        assert cited.references[0].relevance_score == pytest.approx(0.92, abs=1e-3)

    def test_empty_docs(self, agent):
        cited = agent.attach_simple_citations("Test.", [])
        assert cited.references == []

    def test_full_response_with_references(self, agent, retrieved_docs):
        cited = agent.attach_simple_citations("Metformin is used for diabetes.", retrieved_docs)
        full = cited.full_response()
        assert "Metformin" in full
        assert "References:" in full
        assert "Diabetes Management" in full  # from metadata title


# ── CitedResponse ──────────────────────────────────────────────────────────────

class TestCitedResponse:
    def test_formatted_references(self):
        citations = [
            Citation(source="S1", pmid="1", title="T1", year="2023", journal="J1", relevance_score=0.9),
            Citation(source="S2", pmid="2", title="T2", year="2022", journal="J2", relevance_score=0.8),
        ]
        cr = CitedResponse(text="Answer.", references=citations, citation_map={1: "S1", 2: "S2"})
        refs = cr.formatted_references()
        assert "[1]" in refs
        assert "[2]" in refs
        assert "T1" in refs
        assert "T2" in refs

    def test_full_response_combines_text_and_refs(self):
        cit = Citation(source="S", pmid="", title="Study", year="2023", journal="NEJM", relevance_score=0.9)
        cr = CitedResponse(text="My answer.", references=[cit], citation_map={1: "S"})
        full = cr.full_response()
        assert "My answer." in full
        assert "Study" in full
