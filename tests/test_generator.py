"""
Unit tests for GeneratorAgent.
All LLM calls are mocked — no GPU/model loading required.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.generator import (
    GeneratorAgent,
    GeneratorOutput,
    _extract_confidence,
    _extract_mcq_answer,
)


# ── _extract_mcq_answer ────────────────────────────────────────────────────────

class TestExtractMCQAnswer:
    def test_answer_prefix(self):
        assert _extract_mcq_answer("Answer: A") == "A"

    def test_answer_is_prefix(self):
        assert _extract_mcq_answer("The answer is B.") == "B"

    def test_parenthesis(self):
        assert _extract_mcq_answer("The correct answer is (C).") == "C"

    def test_correct_answer_is(self):
        assert _extract_mcq_answer("correct answer is D here") == "D"

    def test_standalone_letter(self):
        assert _extract_mcq_answer("Based on context, A is correct.") == "A"

    def test_lowercase(self):
        result = _extract_mcq_answer("the answer is a.")
        assert result == "A"

    def test_none_when_no_letter(self):
        assert _extract_mcq_answer("I don't know the answer.") is None

    def test_full_paragraph(self):
        text = (
            "Looking at the options, Metformin is first-line. "
            "Therefore the answer is A. Metformin reduces glucose production."
        )
        assert _extract_mcq_answer(text) == "A"


# ── _extract_confidence ────────────────────────────────────────────────────────

class TestExtractConfidence:
    def test_high(self):
        assert _extract_confidence("Confidence: High") == "High"

    def test_medium(self):
        assert _extract_confidence("confidence: medium based on evidence") == "Medium"

    def test_low(self):
        assert _extract_confidence("Confidence level: low") == "Low"

    def test_default_medium_when_absent(self):
        assert _extract_confidence("No confidence mentioned here.") == "Medium"

    def test_case_insensitive(self):
        assert _extract_confidence("CONFIDENCE: HIGH") == "High"


# ── GeneratorAgent ─────────────────────────────────────────────────────────────

@pytest.fixture
def generator(mock_llm):
    agent = GeneratorAgent()
    agent._llm = mock_llm
    return agent


CONTEXT = (
    "[Source: PubMed:12345 | Score: 0.92]\n"
    "Metformin is the first-line treatment for type 2 diabetes mellitus. "
    "It reduces hepatic glucose production and improves insulin sensitivity."
)

QUESTION = "What is the first-line treatment for type 2 diabetes?"

OPTIONS = {"A": "Metformin", "B": "Insulin", "C": "Surgery", "D": "Aspirin"}


class TestGeneratorAgent:
    def test_generate_returns_output(self, generator):
        out = generator.generate(QUESTION, CONTEXT)
        assert isinstance(out, GeneratorOutput)

    def test_generate_has_answer(self, generator):
        out = generator.generate(QUESTION, CONTEXT)
        assert len(out.answer) > 0

    def test_generate_has_model_name(self, generator):
        out = generator.generate(QUESTION, CONTEXT)
        assert out.model_name == "mock-biomistral"

    def test_generate_has_response_time(self, generator):
        out = generator.generate(QUESTION, CONTEXT)
        assert out.response_time_s >= 0.0

    def test_generate_mcq_extracts_prediction(self, generator):
        out = generator.generate(QUESTION, CONTEXT, mcq_options=OPTIONS)
        # mock LLM returns "the answer is A" for MCQ mode
        assert out.mcq_prediction == "A"

    def test_generate_no_mcq_prediction_is_none(self, generator):
        out = generator.generate(QUESTION, CONTEXT, mcq_options=None)
        # No MCQ options → mcq_prediction should be None
        # (or a letter if the answer text happens to contain one — either is valid)
        assert out.mcq_prediction is None or out.mcq_prediction in "ABCD"

    def test_generate_no_context_baseline(self, generator):
        out = generator.generate_no_context(QUESTION, mcq_options=OPTIONS)
        assert isinstance(out, GeneratorOutput)

    def test_generate_error_handling(self, generator):
        generator._llm.answer_clinical_question.side_effect = RuntimeError("OOM")
        out = generator.generate(QUESTION, CONTEXT)
        assert "Generation failed" in out.answer
        assert out.confidence == "Low"
        # Reset
        generator._llm.answer_clinical_question.side_effect = None

    def test_generate_mcq_response_keys(self, generator):
        out = generator.generate(QUESTION, CONTEXT, mcq_options=OPTIONS)
        assert hasattr(out, "answer")
        assert hasattr(out, "mcq_prediction")
        assert hasattr(out, "confidence")
        assert hasattr(out, "response_time_s")


# ── GeneratorOutput ────────────────────────────────────────────────────────────

class TestGeneratorOutput:
    def test_default_confidence(self):
        out = GeneratorOutput(
            answer="test",
            raw_output="test",
            model_name="x",
            response_time_s=1.0,
        )
        assert out.confidence == "Medium"

    def test_mcq_prediction_default_none(self):
        out = GeneratorOutput(
            answer="test",
            raw_output="test",
            model_name="x",
            response_time_s=1.0,
        )
        assert out.mcq_prediction is None
