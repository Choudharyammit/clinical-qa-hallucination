"""
RerankAgent — cross-encoder reranking of retrieved documents.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 to score each (query, document)
pair with a fine-grained relevance score, yielding better ranking than
embedding similarity alone.
"""

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from src.config import settings


@dataclass
class RankedDocument:
    text: str
    source: str
    cross_encoder_score: float
    original_rank: int
    new_rank: int
    metadata: dict


class RerankAgent:
    """
    Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.
    Input: query + list of candidate documents from RetrieverAgent.
    Output: same documents re-ranked by cross-encoder relevance score.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
        device: str = "cpu",
    ):
        self.model_name = model_name or settings.reranker_model
        self.top_k = top_k or settings.reranker_top_k
        self._model = None
        self.device = device

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder reranker: {self.model_name}")
            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.success(f"Cross-encoder loaded: {self.model_name}")

    def rerank(
        self,
        query: str,
        documents: list,  # list[RetrievedDocument | dict]
        top_k: Optional[int] = None,
    ) -> list[RankedDocument]:
        """
        Rerank documents by cross-encoder relevance to query.
        Returns top_k RankedDocument objects sorted by score descending.
        """
        self._ensure_loaded()
        k = top_k or self.top_k

        if not documents:
            return []

        # Normalize input to (text, source, metadata) tuples
        pairs = []
        doc_data = []
        for i, doc in enumerate(documents):
            if hasattr(doc, "text"):
                text, source, meta = doc.text, doc.source, getattr(doc, "metadata", {})
            else:
                text = doc.get("text", "")
                source = doc.get("source", "")
                meta = doc.get("metadata", {})
            pairs.append([query, text])
            doc_data.append((i, text, source, meta))

        # Batch score all pairs
        scores = self._model.predict(pairs, show_progress_bar=False)

        ranked = sorted(
            zip(scores, doc_data),
            key=lambda x: x[0],
            reverse=True,
        )[:k]

        result = []
        for new_rank, (score, (orig_rank, text, source, meta)) in enumerate(ranked, start=1):
            result.append(RankedDocument(
                text=text,
                source=source,
                cross_encoder_score=float(score),
                original_rank=orig_rank + 1,
                new_rank=new_rank,
                metadata=meta,
            ))

        logger.debug(f"Reranked {len(documents)} docs → top {len(result)} | "
                     f"Top score: {result[0].cross_encoder_score:.3f}" if result else "")
        return result

    def rerank_for_context(
        self,
        query: str,
        documents: list,
        max_chunks: Optional[int] = None,
    ) -> tuple[str, list[RankedDocument]]:
        """Rerank and return formatted context string for LLM."""
        max_c = max_chunks or settings.max_context_chunks
        ranked = self.rerank(query, documents, top_k=max_c)

        context_parts = [
            f"[Source: {d.source} | Score: {d.cross_encoder_score:.3f}]\n{d.text}"
            for d in ranked
        ]
        return "\n\n---\n\n".join(context_parts), ranked
