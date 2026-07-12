"""
FastAPI application entry point for the Clinical QA System.

Endpoints:
  POST /query   — clinical question → grounded answer + citations + hallucination score
  GET  /health  — system health
  GET  /metrics — current benchmark stats
  GET  /docs    — Swagger UI (auto-generated)
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src.api.routes import router
from src.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan manager.
    Initialises the orchestrator on startup so the heavy models
    are loaded once and reused across requests.
    """
    logger.info("Starting Clinical QA System…")
    settings.ensure_dirs()

    try:
        from src.agents.orchestrator import OrchestratorAgent
        orchestrator = OrchestratorAgent()
        app.state.orchestrator = orchestrator
        logger.success("OrchestratorAgent ready.")
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        app.state.orchestrator = None

    yield

    logger.info("Shutting down Clinical QA System.")


app = FastAPI(
    title="Clinical QA System",
    description=(
        "Hallucination-Resistant Clinical Question-Answering System "
        "using RAG and Fine-tuned LLMs. "
        "LJMU MSc ML&AI Research Project — Amit Choudhary (1187247)."
    ),
    version="1.0.0",
    contact={
        "name": "Amit Choudhary",
        "email": "1187247@student.ljmu.ac.uk",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Request timing middleware ────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.3f}s"
    return response


# ── Global exception handler ─────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc), "status_code": 500},
    )


# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(router, prefix="")


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
