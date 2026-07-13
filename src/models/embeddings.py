"""
Embedding model wrapper using BioBERT / PubMedBERT via sentence-transformers.
Provides a consistent interface for encoding text to dense vectors.
"""

from typing import Optional, Union

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import settings


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer for biomedical text encoding.
    Uses BioBERT fine-tuned on medical NLI tasks for domain-appropriate embeddings.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        self.batch_size = settings.embedding_batch_size
        logger.info(f"Loading embedding model: {self.model_name}")
        self._model = SentenceTransformer(self.model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()
        logger.success(f"Embedding model ready. Dimension: {self.dimension}")

    def encode(
        self,
        texts: Union[str, list[str]],
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode one or more texts to dense vectors.
        Returns shape (N, dimension) float32 array.
        """
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string. Returns shape (dimension,)."""
        return self.encode(query, normalize=True)[0]

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two already-normalized vectors."""
        return float(np.dot(a, b))

    def batch_similarity(self, query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        """Cosine similarity of query against all corpus vectors. Returns shape (N,)."""
        return corpus @ query


_model_instance: Optional[EmbeddingModel] = None


def get_embedding_model() -> EmbeddingModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = EmbeddingModel()
    return _model_instance
