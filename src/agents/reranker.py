"""
RerankAgent — cross-encoder reranking of retrieved documents.

Uses AutoModelForSequenceClassification directly (avoids sentence_transformers
CrossEncoder subprocess issues on macOS with Python 3.13 + transformers 5.x).
Falls back to TinyBERT → BioBERT similarity if model loading fails.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import settings


@dataclass
class RankedDocument:
    text: str
    source: str
    cross_encoder_score: float
    original_rank: int
    new_rank: int
    metadata: dict


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class RerankAgent:
    """
    Cross-encoder reranker using ms-marco-TinyBERT-L-2-v2 via AutoModel.
    Scores (query, document) pairs using direct HuggingFace inference.
    Falls back to embedding-similarity ordering if model unavailable.
    """

    _CANDIDATES = [
        "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        "cross-encoder/ms-marco-MiniLM-L-2-v2",
    ]

    def __init__(
        self,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or settings.reranker_model
        self.top_k = top_k or settings.reranker_top_k
        self._model = None
        self._tokenizer = None
        self._device = device or _best_device()
        self._fallback_to_similarity = False

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._fallback_to_similarity:
            return

        candidates = [self.model_name] + [
            c for c in self._CANDIDATES if c != self.model_name
        ]
        for model_id in candidates:
            try:
                logger.info(f"Loading cross-encoder: {model_id} on {self._device}")
                tok = AutoTokenizer.from_pretrained(model_id)
                mdl = AutoModelForSequenceClassification.from_pretrained(
                    model_id, dtype=torch.float32
                )
                mdl = mdl.to(self._device)
                mdl.eval()

                # Probe for NaN
                probe_enc = tok(
                    "test query", "test document", return_tensors="pt", truncation=True
                )
                probe_enc = {k: v.to(self._device) for k, v in probe_enc.items()}
                with torch.no_grad():
                    probe_logit = mdl(**probe_enc).logits[0, 0].item()
                if math.isnan(probe_logit) or math.isinf(probe_logit):
                    logger.warning(f"{model_id} probe returned NaN/Inf — skipping")
                    continue

                self._model = mdl
                self._tokenizer = tok
                self.model_name = model_id
                logger.success(f"Cross-encoder loaded: {model_id} on {self._device}")
                return
            except Exception as e:
                logger.warning(f"Could not load {model_id}: {e}")

        logger.warning("No cross-encoder available; falling back to embedding similarity order")
        self._fallback_to_similarity = True

    def _score_pairs(self, query: str, texts: list[str]) -> list[float]:
        """Score (query, text) pairs. Returns raw logit scores."""
        scores = []
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = self._tokenizer(
                [query] * len(batch_texts),
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self._model(**enc).logits
            # Single-class models: logits shape (batch, 1) or (batch,)
            if logits.ndim == 2 and logits.shape[1] == 1:
                batch_scores = logits[:, 0].cpu().tolist()
            elif logits.ndim == 2:
                # Multi-class: use the "relevant" class (index -1)
                batch_scores = logits[:, -1].cpu().tolist()
            else:
                batch_scores = logits.cpu().tolist()
            scores.extend(batch_scores)
        return scores

    def rerank(
        self,
        query: str,
        documents: list,
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

        # Normalize input
        doc_data = []
        for i, doc in enumerate(documents):
            if hasattr(doc, "text"):
                text = doc.text
                source = doc.source
                meta = getattr(doc, "metadata", {})
                sim_score = getattr(doc, "score", 0.0)
            else:
                text = doc.get("text", "")
                source = doc.get("source", "")
                meta = doc.get("metadata", {})
                sim_score = doc.get("score", 0.0)
            doc_data.append((i, text, source, meta, sim_score))

        if self._fallback_to_similarity:
            # Preserve embedding similarity order
            scores = [d[4] for d in doc_data]
        else:
            texts = [d[1] for d in doc_data]
            raw = self._score_pairs(query, texts)
            scores = [0.0 if (math.isnan(s) or math.isinf(s)) else s for s in raw]

        ranked = sorted(
            zip(scores, doc_data),
            key=lambda x: x[0],
            reverse=True,
        )[:k]

        result = []
        for new_rank, (score, (orig_rank, text, source, meta, _)) in enumerate(ranked, start=1):
            result.append(RankedDocument(
                text=text,
                source=source,
                cross_encoder_score=float(score),
                original_rank=orig_rank + 1,
                new_rank=new_rank,
                metadata=meta,
            ))

        if result:
            logger.debug(
                f"Reranked {len(documents)} docs → top {len(result)} | "
                f"Top score: {result[0].cross_encoder_score:.3f}"
            )
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
