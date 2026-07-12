"""
FastAPI route handlers for the Clinical QA System.
"""

import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status
from loguru import logger

from src.api.schemas import (
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
)
from src.config import settings

router = APIRouter()


def _get_orchestrator(request: Request):
    """Retrieve the orchestrator from app state."""
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator not initialized. Check startup logs.",
        )
    return orchestrator


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Answer a clinical question",
    description=(
        "Submit a clinical question and receive an evidence-grounded answer "
        "with citations and hallucination confidence scores."
    ),
    tags=["Clinical QA"],
)
async def query_endpoint(request: Request, body: QueryRequest) -> QueryResponse:
    """
    Main clinical QA endpoint.

    - Retrieves relevant medical evidence from ChromaDB
    - Generates answer via BioMistral-7B-DARE
    - Detects hallucinations via DeBERTa NLI
    - Attaches source citations
    """
    orchestrator = _get_orchestrator(request)

    if len(body.question) > settings.api_max_query_length:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Question exceeds maximum length of {settings.api_max_query_length} characters.",
        )

    logger.info(f"Query: '{body.question[:80]}…'")

    try:
        result = orchestrator.run(
            query=body.question,
            mcq_options=body.mcq_options,
            use_rag=body.use_rag,
            use_hallucination_filter=body.use_hallucination_filter,
        )
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}",
        )

    return QueryResponse(**result)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    tags=["System"],
)
async def health_check(request: Request) -> HealthResponse:
    """Returns system health status including vector store and model state."""
    try:
        from src.data.vectorstore import get_chroma_client, get_or_create_collection
        client = get_chroma_client()
        collection = get_or_create_collection(client)
        vs_count = collection.count()
    except Exception:
        vs_count = -1

    orchestrator = getattr(request.app.state, "orchestrator", None)

    return HealthResponse(
        status="healthy" if orchestrator is not None else "degraded",
        vector_store_count=vs_count,
        model_loaded=orchestrator is not None,
        version="1.0.0",
    )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Current benchmark metrics",
    tags=["Evaluation"],
)
async def metrics_endpoint(request: Request) -> MetricsResponse:
    """Returns the most recent benchmark evaluation metrics from MLflow."""
    import json
    from pathlib import Path

    metrics_file = Path("./data/processed/latest_metrics.json")
    if metrics_file.exists():
        with open(metrics_file) as f:
            data = json.load(f)
        return MetricsResponse(**data)

    return MetricsResponse(
        experiment_name=settings.mlflow_experiment_name,
        accuracy=None,
        hallucination_rate=None,
        avg_response_time_s=None,
        total_questions=None,
        last_updated=None,
    )
