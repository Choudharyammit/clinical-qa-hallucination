"""
OrchestratorAgent — ReAct-style clinical query orchestration.

Implements a Reason → Act → Observe loop (max 5 iterations) to coordinate
all specialist agents into a single coherent clinical QA pipeline.

Shared state dict passed through the pipeline:
{
    query, mcq_options,
    retrieved_docs, reranked_docs, context,
    generator_output, hallucination_report, cited_response,
    final_answer, confidence, response_time_s,
    iterations, errors
}
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger

from src.agents.citation import CitationAgent
from src.agents.generator import GeneratorAgent, GeneratorOutput
from src.agents.hallucination import HallucinationDetectorAgent, HallucinationReport
from src.agents.reranker import RerankAgent
from src.agents.retriever import RetrieverAgent
from src.config import settings


@dataclass
class PipelineState:
    """Shared state passed between all agents in the pipeline."""
    query: str
    mcq_options: Optional[dict[str, str]] = None
    use_rag: bool = True
    use_hallucination_filter: bool = True

    # Filled by agents
    retrieved_docs: list = field(default_factory=list)
    reranked_docs: list = field(default_factory=list)
    context: str = ""
    generator_output: Optional[GeneratorOutput] = None
    hallucination_report: Optional[HallucinationReport] = None
    cited_response: Any = None

    # Final output
    final_answer: str = ""
    mcq_prediction: Optional[str] = None
    confidence: str = "Medium"
    response_time_s: float = 0.0
    hallucination_rate: float = 0.0

    # Diagnostics
    iterations: int = 0
    errors: list[str] = field(default_factory=list)
    agent_times: dict[str, float] = field(default_factory=dict)

    def to_api_response(self) -> dict:
        return {
            "answer": self.final_answer,
            "mcq_prediction": self.mcq_prediction,
            "confidence": self.confidence,
            "hallucination_rate": round(self.hallucination_rate, 4),
            "response_time_s": round(self.response_time_s, 3),
            "citations": [
                {
                    "source": c.source,
                    "title": c.title,
                    "pmid": c.pmid,
                    "year": c.year,
                    "relevance": round(c.relevance_score, 3),
                }
                for c in (self.cited_response.references if self.cited_response else [])
            ],
            "hallucination_report": (
                self.hallucination_report.summary()
                if self.hallucination_report else {}
            ),
            "sources_retrieved": len(self.retrieved_docs),
            "iterations": self.iterations,
        }


class OrchestratorAgent:
    """
    ReAct-style orchestrator for the clinical QA pipeline.

    ReAct loop:
      Reason: Decide which agents to invoke based on current state
      Act:    Invoke the next agent
      Observe: Check output quality; retry if confidence is low
    """

    MAX_ITERATIONS = 5

    def __init__(
        self,
        retriever: Optional[RetrieverAgent] = None,
        reranker: Optional[RerankAgent] = None,
        generator: Optional[GeneratorAgent] = None,
        hallucination_detector: Optional[HallucinationDetectorAgent] = None,
        citation_agent: Optional[CitationAgent] = None,
    ):
        self.retriever = retriever or RetrieverAgent()
        self.reranker = reranker or RerankAgent()
        self.generator = generator or GeneratorAgent()
        self.hallucination_detector = (
            hallucination_detector or HallucinationDetectorAgent()
        )
        self.citation_agent = citation_agent or CitationAgent()

    def run(
        self,
        query: str,
        mcq_options: Optional[dict[str, str]] = None,
        use_rag: bool = True,
        use_hallucination_filter: bool = True,
    ) -> dict:
        """
        Execute the full clinical QA pipeline for a query.
        Returns API-ready response dict.
        """
        wall_start = time.perf_counter()
        state = PipelineState(
            query=query,
            mcq_options=mcq_options,
            use_rag=use_rag,
            use_hallucination_filter=use_hallucination_filter,
        )

        try:
            self._react_loop(state)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            state.errors.append(str(e))
            state.final_answer = f"System error: {e}"
            state.confidence = "Low"

        state.response_time_s = time.perf_counter() - wall_start
        return state.to_api_response()

    def _react_loop(self, state: PipelineState) -> None:
        """
        Main ReAct loop. Each iteration either:
        - Advances to the next pipeline stage, or
        - Retries retrieval with a reformulated query if quality is poor
        """
        for iteration in range(self.MAX_ITERATIONS):
            state.iterations = iteration + 1
            action = self._reason(state)
            logger.debug(f"Iteration {iteration+1}: action={action}")

            if action == "retrieve":
                self._act_retrieve(state)
            elif action == "rerank":
                self._act_rerank(state)
            elif action == "generate":
                self._act_generate(state)
            elif action == "verify":
                self._act_verify(state)
            elif action == "cite":
                self._act_cite(state)
            elif action == "done":
                break
            elif action == "retry_retrieve":
                self._act_retrieve(state, expand_query=True)
            else:
                break

        if not state.final_answer:
            state.final_answer = (
                state.generator_output.answer
                if state.generator_output
                else "Unable to generate answer."
            )
            state.confidence = "Low"

    def _reason(self, state: PipelineState) -> str:
        """
        Decide what action to take next based on current pipeline state.
        This is the Reason step of ReAct.
        """
        if state.use_rag and not state.retrieved_docs:
            return "retrieve"

        if state.use_rag and state.retrieved_docs and not state.reranked_docs:
            return "rerank"

        if state.generator_output is None:
            return "generate"

        if state.use_rag and state.use_hallucination_filter and state.hallucination_report is None:
            return "verify"

        # If hallucination rate is very high (>80%) and we haven't retried yet
        if (
            state.hallucination_report
            and state.hallucination_report.hallucination_rate > 0.8
            and state.iterations < 3
        ):
            return "retry_retrieve"

        if state.cited_response is None:
            return "cite"

        return "done"

    # ── Agent Actions ─────────────────────────────────────────────────────────

    def _act_retrieve(self, state: PipelineState, expand_query: bool = False) -> None:
        t = time.perf_counter()
        query = state.query
        if expand_query:
            # Simple query expansion: add medical context keywords
            query = f"clinical {query} medical treatment diagnosis"

        docs = self.retriever.retrieve(query, top_k=settings.retriever_top_k)
        state.retrieved_docs = docs
        state.agent_times["retriever"] = time.perf_counter() - t
        logger.debug(f"Retrieved {len(docs)} docs in {state.agent_times['retriever']:.2f}s")

    def _act_rerank(self, state: PipelineState) -> None:
        t = time.perf_counter()
        ranked = self.reranker.rerank(
            state.query,
            state.retrieved_docs,
            top_k=settings.max_context_chunks,
        )
        state.reranked_docs = ranked

        # Build context string from reranked docs
        context_parts = [
            f"[Source: {d.source} | Score: {d.cross_encoder_score:.3f}]\n{d.text}"
            for d in ranked
        ]
        state.context = "\n\n---\n\n".join(context_parts)
        state.agent_times["reranker"] = time.perf_counter() - t

    def _act_generate(self, state: PipelineState) -> None:
        t = time.perf_counter()
        context = state.context or "No context available."
        output = self.generator.generate(
            question=state.query,
            context=context,
            mcq_options=state.mcq_options,
        )
        state.generator_output = output
        state.mcq_prediction = output.mcq_prediction
        state.confidence = output.confidence
        state.agent_times["generator"] = time.perf_counter() - t

    def _act_verify(self, state: PipelineState) -> None:
        t = time.perf_counter()
        evidence_texts = [
            d.text for d in (state.reranked_docs or state.retrieved_docs)
        ]
        evidence_sources = [
            d.source for d in (state.reranked_docs or state.retrieved_docs)
        ]
        report = self.hallucination_detector.verify(
            response=state.generator_output.answer,
            evidence_chunks=evidence_texts,
            evidence_sources=evidence_sources,
        )
        state.hallucination_report = report
        state.hallucination_rate = report.hallucination_rate
        state.agent_times["hallucination"] = time.perf_counter() - t
        logger.debug(f"Hallucination rate: {report.hallucination_rate:.2%}")

    def _act_cite(self, state: PipelineState) -> None:
        t = time.perf_counter()
        response_text = (
            state.hallucination_report.verified_response
            if state.hallucination_report
            else state.generator_output.answer
        )
        docs = state.reranked_docs or state.retrieved_docs

        if state.hallucination_report:
            cited = self.citation_agent.attach_citations(
                response=response_text,
                hallucination_report=state.hallucination_report,
                retrieved_docs=docs,
            )
        else:
            cited = self.citation_agent.attach_simple_citations(
                response=response_text,
                retrieved_docs=docs,
            )

        state.cited_response = cited
        state.final_answer = cited.full_response()
        state.agent_times["citation"] = time.perf_counter() - t
