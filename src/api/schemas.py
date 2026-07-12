"""
Pydantic schemas for the Clinical QA REST API.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Clinical question to answer",
        examples=["What are the first-line treatments for type 2 diabetes mellitus?"],
    )
    mcq_options: Optional[dict[str, str]] = Field(
        default=None,
        description="MCQ options A-D (optional, for USMLE-style questions)",
        examples=[{"A": "Metformin", "B": "Insulin", "C": "Glipizide", "D": "Sitagliptin"}],
    )
    use_rag: bool = Field(default=True, description="Enable RAG retrieval")
    use_hallucination_filter: bool = Field(
        default=True, description="Enable post-generation hallucination detection"
    )
    max_context_chunks: Optional[int] = Field(
        default=None, ge=1, le=20,
        description="Override number of context chunks to retrieve",
    )

    @field_validator("question")
    @classmethod
    def clean_question(cls, v: str) -> str:
        return v.strip()


class CitationResponse(BaseModel):
    source: str
    title: str
    pmid: str
    year: str
    relevance: float


class HallucinationSummary(BaseModel):
    total_claims: int
    supported_claims: int
    hallucinated_claims: int
    hallucination_rate: float
    overall_confidence: float = 0.0


class QueryResponse(BaseModel):
    answer: str = Field(description="Full answer text with inline citations")
    mcq_prediction: Optional[str] = Field(
        default=None, description="Predicted MCQ letter (A/B/C/D) if options provided"
    )
    confidence: str = Field(description="Confidence level: High / Medium / Low")
    hallucination_rate: float = Field(description="Fraction of claims not supported by evidence")
    response_time_s: float = Field(description="End-to-end response time in seconds")
    citations: list[CitationResponse] = Field(
        default_factory=list,
        description="Source citations for factual claims",
    )
    hallucination_report: dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed hallucination detection report",
    )
    sources_retrieved: int = Field(description="Number of evidence chunks retrieved")
    iterations: int = Field(description="ReAct loop iterations used")


class HealthResponse(BaseModel):
    status: str
    vector_store_count: int
    model_loaded: bool
    version: str = "1.0.0"


class MetricsResponse(BaseModel):
    experiment_name: str
    accuracy: Optional[float] = None
    hallucination_rate: Optional[float] = None
    avg_response_time_s: Optional[float] = None
    total_questions: Optional[int] = None
    last_updated: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    detail: str
    status_code: int
