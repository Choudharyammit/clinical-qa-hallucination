"""
Centralized model manager for the Clinical QA System.

Provides a single, lazy-loaded, cached access point for all heavy models:
  - LLM (BioMistral / Ollama / BioGPT fallback)
  - Embedding model (BioBERT)
  - NLI model (DeBERTa-v3)
  - Reranker (cross-encoder)

Why: prevents reloading models on each agent instantiation and ensures
all agents share the same in-memory model instance (critical for GPU memory).

Usage:
    from src.models.model_manager import model_manager
    llm = model_manager.llm
    embed = model_manager.embedding_model
    nli  = model_manager.nli_model
"""

import threading
from typing import Optional

from loguru import logger

from src.config import settings


class ModelManager:
    """
    Thread-safe lazy singleton for all ML models.
    Models are loaded on first access and held for the process lifetime.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._llm = None
        self._embedding_model = None
        self._nli_model = None
        self._reranker = None
        self._llm_prefer: str = "biomistral"

    # ── LLM ──────────────────────────────────────────────────────────────────

    def set_llm_preference(self, prefer: str) -> None:
        """Set preferred LLM backend before first load ('biomistral', 'ollama', 'biogpt')."""
        self._llm_prefer = prefer

    @property
    def llm(self):
        if self._llm is None:
            with self._lock:
                if self._llm is None:
                    from src.models.llm import get_llm
                    logger.info(f"Loading LLM (prefer={self._llm_prefer})…")
                    self._llm = get_llm(prefer=self._llm_prefer)
        return self._llm

    def set_llm(self, llm) -> None:
        """Inject a pre-loaded LLM (e.g. a mock in tests)."""
        with self._lock:
            self._llm = llm

    # ── Embeddings ────────────────────────────────────────────────────────────

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            with self._lock:
                if self._embedding_model is None:
                    from src.models.embeddings import EmbeddingModel
                    logger.info("Loading embedding model…")
                    self._embedding_model = EmbeddingModel()
        return self._embedding_model

    def embedding_fn(self):
        """Return a plain function for encode() — compatible with vectorstore API."""
        model = self.embedding_model
        return lambda texts: model.encode(texts, normalize=True).tolist()

    # ── NLI model ─────────────────────────────────────────────────────────────

    @property
    def nli_model(self):
        if self._nli_model is None:
            with self._lock:
                if self._nli_model is None:
                    from src.models.nli import MedNLIModel
                    logger.info("Loading NLI model…")
                    self._nli_model = MedNLIModel()
                    self._nli_model.load()
        return self._nli_model

    def set_nli_model(self, nli) -> None:
        with self._lock:
            self._nli_model = nli

    # ── Reranker ──────────────────────────────────────────────────────────────

    @property
    def reranker(self):
        if self._reranker is None:
            with self._lock:
                if self._reranker is None:
                    from sentence_transformers import CrossEncoder
                    logger.info(f"Loading cross-encoder reranker: {settings.reranker_model}")
                    self._reranker = CrossEncoder(settings.reranker_model, device="cpu")
        return self._reranker

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return which models are currently loaded."""
        return {
            "llm": self._llm is not None,
            "llm_model": getattr(self._llm, "model_name", None) if self._llm else None,
            "embedding_model": self._embedding_model is not None,
            "nli_model": self._nli_model is not None,
            "reranker": self._reranker is not None,
        }

    def unload_all(self) -> None:
        """Release all models from memory (useful between benchmark runs)."""
        import gc
        with self._lock:
            self._llm = None
            self._embedding_model = None
            self._nli_model = None
            self._reranker = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("All models unloaded.")


# Global singleton
model_manager = ModelManager()
