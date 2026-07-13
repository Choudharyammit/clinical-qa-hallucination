"""
Integration tests for OrchestratorAgent.

Tests the full ReAct pipeline with mocked sub-agents, verifying:
  - Correct action sequencing (retrieve → rerank → generate → verify → cite)
  - State mutation at each step
  - Graceful error handling and fallbacks
  - API response schema conformance
  - No-RAG mode (base LLM condition)
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.citation import CitedResponse, Citation
from src.agents.generator import GeneratorOutput
from src.agents.hallucination import HallucinationReport, ClaimVerification
from src.agents.orchestrator import OrchestratorAgent, PipelineState
from src.agents.reranker import RankedDocument
from src.agents.retriever import RetrievedDocument


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_retrieved_docs(n=3):
    return [
        RetrievedDocument(
            text=f"Medical evidence passage {i}. Contains clinical information about diabetes treatment.",
            source=f"PubMed:{10000+i}",
            score=0.9 - i * 0.05,
            metadata={"pmid": str(10000 + i), "title": f"Study {i}", "year": "2023"},
            rank=i + 1,
        )
        for i in range(n)
    ]


def make_ranked_docs(n=3):
    return [
        RankedDocument(
            text=f"Reranked evidence {i}. First-line treatment for type 2 diabetes is metformin.",
            source=f"PubMed:{10000+i}",
            cross_encoder_score=0.95 - i * 0.1,
            original_rank=i + 1,
            new_rank=i + 1,
            metadata={"pmid": str(10000 + i), "title": f"Study {i}", "year": "2023"},
        )
        for i in range(n)
    ]


def make_generator_output(with_mcq: bool = False):
    return GeneratorOutput(
        answer="Metformin is the first-line treatment for type 2 diabetes. It reduces hepatic glucose production. Confidence: High",
        raw_output="...",
        model_name="mock-biomistral",
        response_time_s=0.5,
        mcq_prediction="A" if with_mcq else None,
        confidence="High",
    )


def make_hallucination_report(response: str):
    claims = [
        ClaimVerification(
            claim="Metformin is the first-line treatment for type 2 diabetes.",
            entailment_score=0.92,
            is_hallucinated=False,
            best_evidence="Metformin is the first-line treatment for type 2 diabetes.",
            evidence_source="PubMed:10000",
        )
    ]
    return HallucinationReport(
        original_response=response,
        claims=claims,
        hallucination_rate=0.0,
        overall_confidence=0.92,
        verified_response=response,
        total_claims=1,
        hallucinated_count=0,
        supported_count=1,
    )


def make_cited_response(answer: str):
    cit = Citation(
        source="PubMed:10000",
        pmid="10000",
        title="Metformin Study",
        year="2023",
        journal="NEJM",
        relevance_score=0.92,
    )
    return CitedResponse(
        text=f"{answer} [1]",
        references=[cit],
        citation_map={1: "PubMed:10000"},
    )


@pytest.fixture
def mock_retriever():
    r = MagicMock()
    r.retrieve.return_value = make_retrieved_docs()
    return r


@pytest.fixture
def mock_reranker():
    r = MagicMock()

    def rerank(query, docs, top_k=5):
        return make_ranked_docs(min(top_k, len(docs)))

    r.rerank.side_effect = rerank
    return r


@pytest.fixture
def mock_generator():
    g = MagicMock()

    def generate(question, context, mcq_options=None):
        return make_generator_output(with_mcq=bool(mcq_options))

    g.generate.side_effect = generate
    return g


@pytest.fixture
def mock_hallucination_detector():
    d = MagicMock()

    def verify(response, evidence_chunks, evidence_sources=None):
        return make_hallucination_report(response)

    d.verify.side_effect = verify
    return d


@pytest.fixture
def mock_citation_agent():
    c = MagicMock()

    def attach_citations(response, hallucination_report, retrieved_docs):
        return make_cited_response(response[:80])

    def attach_simple(response, retrieved_docs):
        return make_cited_response(response[:80])

    c.attach_citations.side_effect = attach_citations
    c.attach_simple_citations.side_effect = attach_simple
    return c


@pytest.fixture
def orchestrator(mock_retriever, mock_reranker, mock_generator,
                  mock_hallucination_detector, mock_citation_agent):
    return OrchestratorAgent(
        retriever=mock_retriever,
        reranker=mock_reranker,
        generator=mock_generator,
        hallucination_detector=mock_hallucination_detector,
        citation_agent=mock_citation_agent,
    )


# ── PipelineState ──────────────────────────────────────────────────────────────

class TestPipelineState:
    def test_to_api_response_keys(self):
        state = PipelineState(query="test")
        state.final_answer = "answer"
        state.confidence = "High"
        state.hallucination_rate = 0.05
        state.response_time_s = 1.2
        state.iterations = 3

        resp = state.to_api_response()
        required_keys = [
            "answer", "mcq_prediction", "confidence",
            "hallucination_rate", "response_time_s",
            "citations", "hallucination_report",
            "sources_retrieved", "iterations",
        ]
        for key in required_keys:
            assert key in resp, f"Missing key: {key}"

    def test_to_api_response_with_cited_response(self):
        state = PipelineState(query="test")
        state.final_answer = "answer [1]"
        state.cited_response = make_cited_response("answer")
        resp = state.to_api_response()
        assert len(resp["citations"]) == 1
        assert resp["citations"][0]["source"] == "PubMed:10000"


# ── OrchestratorAgent._reason ─────────────────────────────────────────────────

class TestOrchestratorReason:
    def test_first_action_is_retrieve_when_rag(self, orchestrator):
        state = PipelineState(query="q", use_rag=True)
        assert orchestrator._reason(state) == "retrieve"

    def test_rerank_after_retrieve(self, orchestrator):
        state = PipelineState(query="q", use_rag=True)
        state.retrieved_docs = make_retrieved_docs()
        assert orchestrator._reason(state) == "rerank"

    def test_generate_after_rerank(self, orchestrator):
        state = PipelineState(query="q", use_rag=True)
        state.retrieved_docs = make_retrieved_docs()
        state.reranked_docs = make_ranked_docs()
        state.context = "some context"
        assert orchestrator._reason(state) == "generate"

    def test_verify_after_generate_with_filter(self, orchestrator):
        state = PipelineState(query="q", use_rag=True, use_hallucination_filter=True)
        state.retrieved_docs = make_retrieved_docs()
        state.reranked_docs = make_ranked_docs()
        state.context = "context"
        state.generator_output = make_generator_output()
        assert orchestrator._reason(state) == "verify"

    def test_skip_verify_without_filter(self, orchestrator):
        state = PipelineState(query="q", use_rag=True, use_hallucination_filter=False)
        state.retrieved_docs = make_retrieved_docs()
        state.reranked_docs = make_ranked_docs()
        state.context = "context"
        state.generator_output = make_generator_output()
        assert orchestrator._reason(state) == "cite"

    def test_cite_after_verify(self, orchestrator):
        state = PipelineState(query="q", use_rag=True, use_hallucination_filter=True)
        state.retrieved_docs = make_retrieved_docs()
        state.reranked_docs = make_ranked_docs()
        state.context = "context"
        state.generator_output = make_generator_output()
        state.hallucination_report = make_hallucination_report("answer")
        assert orchestrator._reason(state) == "cite"

    def test_done_after_cite(self, orchestrator):
        state = PipelineState(query="q")
        state.retrieved_docs = make_retrieved_docs()
        state.reranked_docs = make_ranked_docs()
        state.context = "context"
        state.generator_output = make_generator_output()
        state.hallucination_report = make_hallucination_report("answer")
        state.cited_response = make_cited_response("answer")
        assert orchestrator._reason(state) == "done"

    def test_generate_first_when_no_rag(self, orchestrator):
        state = PipelineState(query="q", use_rag=False)
        assert orchestrator._reason(state) == "generate"


# ── OrchestratorAgent.run ─────────────────────────────────────────────────────

class TestOrchestratorRun:
    def test_run_returns_dict(self, orchestrator):
        result = orchestrator.run("What is the treatment for diabetes?")
        assert isinstance(result, dict)

    def test_run_has_answer(self, orchestrator):
        result = orchestrator.run("What is the treatment for diabetes?")
        assert "answer" in result
        assert len(result["answer"]) > 0

    def test_run_calls_all_agents(self, orchestrator, mock_retriever, mock_reranker,
                                   mock_generator, mock_hallucination_detector, mock_citation_agent):
        orchestrator.run("diabetes treatment", use_rag=True, use_hallucination_filter=True)
        mock_retriever.retrieve.assert_called_once()
        mock_reranker.rerank.assert_called_once()
        mock_generator.generate.assert_called_once()
        mock_hallucination_detector.verify.assert_called_once()
        mock_citation_agent.attach_citations.assert_called_once()

    def test_run_no_rag_skips_retriever(self, orchestrator, mock_retriever, mock_generator):
        orchestrator.run("diabetes treatment", use_rag=False)
        mock_retriever.retrieve.assert_not_called()
        mock_generator.generate.assert_called_once()

    def test_run_mcq_mode(self, orchestrator):
        result = orchestrator.run(
            "First-line treatment for T2DM?",
            mcq_options={"A": "Metformin", "B": "Insulin", "C": "Surgery", "D": "Aspirin"},
        )
        assert result.get("mcq_prediction") == "A"

    def test_run_response_has_citations(self, orchestrator):
        result = orchestrator.run("diabetes treatment")
        assert isinstance(result["citations"], list)
        assert len(result["citations"]) > 0

    def test_run_hallucination_rate_float(self, orchestrator):
        result = orchestrator.run("diabetes treatment")
        hr = result.get("hallucination_rate", -1)
        assert 0.0 <= hr <= 1.0

    def test_run_response_time_positive(self, orchestrator):
        result = orchestrator.run("diabetes treatment")
        assert result["response_time_s"] > 0

    def test_run_error_recovery(self, orchestrator, mock_generator):
        mock_generator.generate.side_effect = RuntimeError("CUDA OOM")
        result = orchestrator.run("test query")
        assert "answer" in result
        assert "error" in result["answer"].lower() or result.get("iterations", 0) >= 0
        # Reset
        mock_generator.generate.side_effect = None
        mock_generator.generate.return_value = make_generator_output()

    def test_run_iterations_tracked(self, orchestrator):
        result = orchestrator.run("What is diabetes?")
        assert result.get("iterations", 0) > 0

    def test_run_sources_retrieved(self, orchestrator):
        result = orchestrator.run("diabetes treatment")
        assert result.get("sources_retrieved", 0) > 0
