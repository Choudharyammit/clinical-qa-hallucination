"""
GeneratorAgent — LLM-based clinical answer generation with context injection.

Wraps the LLM models from src.models.llm with:
  - Structured clinical prompt building
  - MCQ answer extraction (for MedQA evaluation)
  - Streaming support placeholder
  - Response post-processing
"""

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from src.config import settings


@dataclass
class GeneratorOutput:
    answer: str
    raw_output: str
    model_name: str
    response_time_s: float
    prompt_tokens: int = 0
    mcq_prediction: Optional[str] = None  # A/B/C/D if MCQ
    confidence: str = "Medium"
    metadata: dict = field(default_factory=dict)


# Regex to extract MCQ letter from model output
_MCQ_RE = re.compile(
    r"\b([A-D])\b(?:\.|:|\)|\s|$)",
    re.IGNORECASE,
)
_CONFIDENCE_RE = re.compile(r"confidence[:\s]+(high|medium|low)", re.IGNORECASE)


def _extract_mcq_answer(text: str) -> Optional[str]:
    """Extract A/B/C/D from model output for MCQ evaluation."""
    # Look for patterns like "Answer: B", "The answer is A", "(B)", etc.
    patterns = [
        r"(?:the\s+)?answer\s*(?:is\s*)?[:\s]*([A-D])\b",
        r"^([A-D])[.\):]",
        r"\b([A-D])\s+is\s+correct",
        r"correct\s+answer\s+is\s+([A-D])",
        r"\(([A-D])\)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()

    # Fallback: first standalone letter in the response
    m = _MCQ_RE.search(text)
    if m:
        return m.group(1).upper()
    return None


def _extract_confidence(text: str) -> str:
    m = _CONFIDENCE_RE.search(text)
    if m:
        return m.group(1).capitalize()
    return "Medium"


class GeneratorAgent:
    """
    Generates clinical answers using an LLM conditioned on retrieved context.
    Supports free-form QA and MCQ (A/B/C/D) formats.
    """

    def __init__(self, llm=None, prefer_llm: str = "biomistral"):
        self._llm = llm
        self._prefer_llm = prefer_llm

    def _ensure_llm(self) -> None:
        if self._llm is None:
            from src.models.llm import get_llm
            logger.info(f"Loading LLM (prefer={self._prefer_llm})…")
            self._llm = get_llm(prefer=self._prefer_llm)

    def generate(
        self,
        question: str,
        context: str,
        mcq_options: Optional[dict[str, str]] = None,
    ) -> GeneratorOutput:
        """
        Generate a clinical answer for a question given retrieval context.

        Args:
            question: Clinical question text
            context: Concatenated retrieved document chunks
            mcq_options: Dict like {'A': 'text', 'B': 'text', ...} for MCQ

        Returns:
            GeneratorOutput with answer, timing, and MCQ prediction
        """
        self._ensure_llm()
        start = time.perf_counter()

        try:
            result = self._llm.answer_clinical_question(
                question=question,
                context=context,
                mcq_options=mcq_options,
            )
            raw = result.get("answer", "")
            elapsed = result.get("response_time_s", time.perf_counter() - start)

            return GeneratorOutput(
                answer=raw,
                raw_output=raw,
                model_name=result.get("model", "unknown"),
                response_time_s=elapsed,
                prompt_tokens=result.get("prompt_tokens", 0),
                mcq_prediction=_extract_mcq_answer(raw) if mcq_options else None,
                confidence=_extract_confidence(raw),
                metadata={"context_length": len(context)},
            )

        except Exception as e:
            logger.error(f"Generator error: {e}")
            return GeneratorOutput(
                answer=f"Generation failed: {str(e)}",
                raw_output="",
                model_name="error",
                response_time_s=time.perf_counter() - start,
                confidence="Low",
            )

    def generate_no_context(self, question: str, mcq_options: Optional[dict] = None) -> GeneratorOutput:
        """Generate answer WITHOUT retrieval (baseline for hallucination comparison)."""
        return self.generate(
            question=question,
            context="No context provided. Answer from parametric knowledge only.",
            mcq_options=mcq_options,
        )
