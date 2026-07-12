"""
Central configuration for the Clinical QA System.
All settings are loaded from environment variables via pydantic-settings.
Import `settings` wherever configuration is needed.
"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────
    llm_model_name: str = "BioMistral/BioMistral-7B-DARE"
    llm_quantization: str = "4bit"
    llm_max_new_tokens: int = 512
    llm_temperature: float = 0.1
    llm_device: str = "auto"

    # ── Embeddings ────────────────────────────────────────────────
    embedding_model: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    embedding_dimension: int = 768
    embedding_batch_size: int = 32

    # ── Vector Store ──────────────────────────────────────────────
    chroma_persist_dir: str = str(ROOT_DIR / "data" / "chroma_db")
    chroma_collection_name: str = "medical_corpus"
    chroma_distance_metric: str = "cosine"

    # ── Reranker ──────────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5

    # ── NLI / Hallucination Detection ─────────────────────────────
    nli_model: str = "microsoft/deberta-v3-base"
    nli_entailment_threshold: float = 0.7
    nli_batch_size: int = 8

    # ── RAG Pipeline ──────────────────────────────────────────────
    retriever_top_k: int = 10
    retriever_mmr_lambda: float = 0.5
    max_context_chunks: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── API ───────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    api_reload: bool = False
    api_max_query_length: int = 1000
    api_rate_limit: int = 60

    # ── MLflow ────────────────────────────────────────────────────
    mlflow_tracking_uri: str = str(ROOT_DIR / "mlruns")
    mlflow_experiment_name: str = "clinical-qa-hallucination"

    # ── PubMed ────────────────────────────────────────────────────
    pubmed_email: str = "1187247@student.ljmu.ac.uk"
    pubmed_max_abstracts: int = 50000
    pubmed_batch_size: int = 200
    huggingface_hub_token: str = ""

    # ── Evaluation ────────────────────────────────────────────────
    eval_batch_size: int = 16
    eval_max_samples: int = 500
    medqa_test_subset: int = 500

    # ── Logging ───────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: str = str(ROOT_DIR / "logs" / "clinical_qa.log")

    # ── Reproducibility ───────────────────────────────────────────
    random_seed: int = 42
    numpy_seed: int = 42
    torch_seed: int = 42

    # ── Derived Paths ─────────────────────────────────────────────
    @property
    def data_raw_dir(self) -> Path:
        return ROOT_DIR / "data" / "raw"

    @property
    def data_processed_dir(self) -> Path:
        return ROOT_DIR / "data" / "processed"

    @property
    def logs_dir(self) -> Path:
        return ROOT_DIR / "logs"

    def ensure_dirs(self) -> None:
        """Create all required runtime directories."""
        for d in [
            self.data_raw_dir,
            self.data_processed_dir,
            Path(self.chroma_persist_dir),
            Path(self.mlflow_tracking_uri),
            self.logs_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
