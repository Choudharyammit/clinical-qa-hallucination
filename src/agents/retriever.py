"""
RetrieverAgent — semantic search over the ChromaDB medical corpus.

Implements:
  - Dense retrieval via BioBERT embeddings + cosine similarity
  - MMR (Maximal Marginal Relevance) for result diversity
  - Recall@K, MRR evaluation helpers
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger

from src.config import settings
from src.data.vectorstore import (
    get_chroma_client,
    get_embedding_function,
    get_or_create_collection,
    mmr_retrieval,
    semantic_search,
)


@dataclass
class RetrievedDocument:
    text: str
    source: str
    score: float
    metadata: dict = field(default_factory=dict)
    rank: int = 0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata,
            "rank": self.rank,
        }


class RetrieverAgent:
    """
    Retrieves relevant medical documents for a given clinical query.

    Retrieval strategy:
      1. Encode query with BioBERT embeddings
      2. Cosine similarity search in ChromaDB (top fetch_k candidates)
      3. MMR re-ranking for diversity if use_mmr=True
      4. Return top_k results with scores
    """

    def __init__(
        self,
        top_k: int = 10,
        fetch_k: int = 30,
        use_mmr: bool = True,
        mmr_lambda: float = 0.5,
        collection_name: Optional[str] = None,
    ):
        self.top_k = top_k or settings.retriever_top_k
        self.fetch_k = fetch_k
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda or settings.retriever_mmr_lambda
        self.collection_name = collection_name or settings.chroma_collection_name

        # Lazy-loaded
        self._embed_fn = None
        self._collection = None

    def _ensure_ready(self) -> None:
        if self._embed_fn is None:
            self._embed_fn = get_embedding_function()
        if self._collection is None:
            client = get_chroma_client()
            self._collection = get_or_create_collection(client, self.collection_name)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_mmr: Optional[bool] = None,
    ) -> list[RetrievedDocument]:
        """
        Retrieve relevant documents for a query.
        Returns ranked list of RetrievedDocument.
        """
        self._ensure_ready()
        k = top_k or self.top_k
        mmr = use_mmr if use_mmr is not None else self.use_mmr

        if self._collection.count() == 0:
            logger.warning("Vector store is empty. Run data ingestion first.")
            return []

        if mmr:
            raw_results = mmr_retrieval(
                query,
                self._collection,
                self._embed_fn,
                top_k=k,
                fetch_k=self.fetch_k,
                lambda_param=self.mmr_lambda,
            )
        else:
            raw_results = semantic_search(
                query,
                self._collection,
                self._embed_fn,
                top_k=k,
            )

        docs = [
            RetrievedDocument(
                text=r["text"],
                source=r["source"],
                score=r["score"],
                metadata=r.get("metadata", {}),
                rank=i + 1,
            )
            for i, r in enumerate(raw_results)
        ]

        logger.debug(f"Retrieved {len(docs)} docs for query: '{query[:60]}…'")
        return docs

    def retrieve_for_context(
        self,
        query: str,
        max_chunks: Optional[int] = None,
    ) -> tuple[str, list[RetrievedDocument]]:
        """
        Retrieve and format context string for LLM prompt injection.
        Returns (context_string, list_of_docs).
        """
        max_c = max_chunks or settings.max_context_chunks
        docs = self.retrieve(query, top_k=max_c)

        context_parts = []
        for doc in docs:
            context_parts.append(
                f"[Source: {doc.source} | Relevance: {doc.score:.2f}]\n{doc.text}"
            )

        context = "\n\n---\n\n".join(context_parts)
        return context, docs

    # ── Evaluation Helpers ────────────────────────────────────────────────────

    def recall_at_k(
        self,
        query: str,
        relevant_pmids: list[str],
        k: int = 10,
    ) -> float:
        """
        Compute Recall@K: fraction of relevant PMIDs found in top-k results.
        relevant_pmids: list of PMIDs known to be relevant for the query.
        """
        docs = self.retrieve(query, top_k=k, use_mmr=False)
        retrieved_pmids = {d.metadata.get("pmid", "") for d in docs}
        relevant_set = set(relevant_pmids)
        if not relevant_set:
            return 0.0
        return len(retrieved_pmids & relevant_set) / len(relevant_set)

    def mean_reciprocal_rank(
        self,
        query: str,
        relevant_pmids: list[str],
        k: int = 10,
    ) -> float:
        """Compute MRR for a single query."""
        docs = self.retrieve(query, top_k=k, use_mmr=False)
        for rank, doc in enumerate(docs, start=1):
            if doc.metadata.get("pmid", "") in relevant_pmids:
                return 1.0 / rank
        return 0.0
