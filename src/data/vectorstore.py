"""
ChromaDB vector store manager for the Clinical QA System.

Handles:
  - Initializing a persistent ChromaDB collection
  - Batch ingestion of processed JSONL chunks with BioBERT embeddings
  - Semantic retrieval (cosine similarity + MMR diversity)
  - FAISS fallback for environments without persistent storage

Usage:
    python -m src.data.vectorstore --action ingest
    python -m src.data.vectorstore --action query --query "diabetes treatment guidelines"
    python -m src.data.vectorstore --action stats
"""

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger
from tqdm import tqdm

from src.config import settings


# ── ChromaDB Client ───────────────────────────────────────────────────────────

def get_chroma_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client."""
    persist_dir = Path(settings.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    logger.debug(f"ChromaDB client initialized at {persist_dir}")
    return client


def get_or_create_collection(
    client: chromadb.PersistentClient,
    collection_name: Optional[str] = None,
) -> chromadb.Collection:
    """Get or create the medical corpus collection."""
    name = collection_name or settings.chroma_collection_name
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": settings.chroma_distance_metric},
    )
    logger.info(f"Collection '{name}' ready. Count: {collection.count()}")
    return collection


# ── Embedding Model ───────────────────────────────────────────────────────────

_embedding_fn_cache: dict[str, Any] = {}


def get_embedding_function(model_name: Optional[str] = None):
    """
    Lazy-load BioBERT sentence-transformers embedding function.
    Cached so model is only loaded once per process.
    """
    model_name = model_name or settings.embedding_model
    if model_name not in _embedding_fn_cache:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)

        def embed_fn(texts: list[str]) -> list[list[float]]:
            return model.encode(
                texts,
                batch_size=settings.embedding_batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).tolist()

        _embedding_fn_cache[model_name] = embed_fn
        logger.success(f"Embedding model loaded: {model_name}")

    return _embedding_fn_cache[model_name]


# ── Ingestion ─────────────────────────────────────────────────────────────────

INGEST_BATCH_SIZE = 256


def ingest_jsonl_file(
    jsonl_path: Path,
    collection: chromadb.Collection,
    embed_fn,
    skip_existing: bool = True,
) -> int:
    """
    Ingest a processed JSONL file into ChromaDB.
    Returns number of new documents added.
    """
    if not jsonl_path.exists():
        logger.warning(f"File not found, skipping: {jsonl_path}")
        return 0

    existing_ids: set[str] = set()
    if skip_existing and collection.count() > 0:
        existing_ids = set(collection.get(include=[])["ids"])

    batch_ids, batch_docs, batch_metas = [], [], []
    added = 0
    skipped = 0

    def flush():
        nonlocal added
        if not batch_ids:
            return
        embeddings = embed_fn(batch_docs)
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=batch_metas,
        )
        added += len(batch_ids)
        batch_ids.clear()
        batch_docs.clear()
        batch_metas.clear()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Ingesting {jsonl_path.name}"):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            chunk_id = record.get("chunk_id", "")
            text = record.get("text", "")
            if not chunk_id or not text:
                continue

            if chunk_id in existing_ids:
                skipped += 1
                continue

            # Build metadata (ChromaDB requires str/int/float/bool values only)
            meta = {
                "source": str(record.get("source", "")),
                "pmid": str(record.get("pmid", "")),
                "title": str(record.get("title", ""))[:200],
                "year": str(record.get("year", "")),
                "journal": str(record.get("journal", ""))[:100],
                "chunk_type": str(record.get("chunk_type", "document")),
                "chunk_index": int(record.get("chunk_index", 0)),
                "pubid": str(record.get("pubid", "")),
                "question": str(record.get("question", ""))[:200],
            }

            batch_ids.append(chunk_id)
            batch_docs.append(text)
            batch_metas.append(meta)

            if len(batch_ids) >= INGEST_BATCH_SIZE:
                flush()

    flush()
    logger.info(f"  Added: {added} | Skipped (existing): {skipped}")
    return added


def ingest_all_processed(collection: chromadb.Collection, embed_fn) -> int:
    """Ingest all processed JSONL files from data/processed/."""
    proc_dir = settings.data_processed_dir
    total = 0
    for jsonl_path in sorted(proc_dir.glob("*.jsonl")):
        # Skip MedQA eval files — these are for evaluation, not retrieval corpus
        # Note: use startswith so "pubmedqa" chunks are NOT skipped
        if jsonl_path.name.startswith("medqa"):
            logger.info(f"Skipping eval file from corpus: {jsonl_path.name}")
            continue
        total += ingest_jsonl_file(jsonl_path, collection, embed_fn)
    logger.success(f"Total new documents ingested: {total}")
    return total


# ── Retrieval ─────────────────────────────────────────────────────────────────

def semantic_search(
    query: str,
    collection: chromadb.Collection,
    embed_fn,
    top_k: int = 10,
    where: Optional[dict] = None,
) -> list[dict]:
    """
    Pure semantic search via cosine similarity.
    Returns list of {text, source, score, metadata} dicts.
    """
    query_embedding = embed_fn([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
        where=where,
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        score = 1.0 - (dist / 2.0)
        hits.append({
            "text": doc,
            "source": meta.get("source", ""),
            "score": round(score, 4),
            "metadata": meta,
        })

    return hits


def mmr_retrieval(
    query: str,
    collection: chromadb.Collection,
    embed_fn,
    top_k: int = 10,
    fetch_k: int = 30,
    lambda_param: float = 0.5,
) -> list[dict]:
    """
    Maximal Marginal Relevance retrieval for diversity.
    Fetches fetch_k candidates, then greedily selects top_k diverse results.
    lambda_param: 0 = max diversity, 1 = max relevance.
    """
    import numpy as np

    candidates = semantic_search(query, collection, embed_fn, top_k=fetch_k)
    if len(candidates) <= top_k:
        return candidates

    query_emb = np.array(embed_fn([query])[0])
    candidate_texts = [c["text"] for c in candidates]
    candidate_embs = np.array(embed_fn(candidate_texts))

    # Normalize
    norms = np.linalg.norm(candidate_embs, axis=1, keepdims=True)
    candidate_embs = candidate_embs / (norms + 1e-8)
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)

    relevance_scores = candidate_embs @ query_norm
    selected_indices: list[int] = []

    for _ in range(top_k):
        if not selected_indices:
            selected_indices.append(int(np.argmax(relevance_scores)))
            continue

        selected_embs = candidate_embs[selected_indices]
        sim_to_selected = candidate_embs @ selected_embs.T
        max_sim_to_selected = sim_to_selected.max(axis=1)

        mmr_scores = (
            lambda_param * relevance_scores
            - (1 - lambda_param) * max_sim_to_selected
        )
        mmr_scores[selected_indices] = -1e9  # Mask already selected
        selected_indices.append(int(np.argmax(mmr_scores)))

    return [candidates[i] for i in selected_indices]


# ── FAISS Fallback ────────────────────────────────────────────────────────────

class FAISSVectorStore:
    """
    Lightweight FAISS-based fallback for environments where ChromaDB is unavailable.
    Stores embeddings in memory; no persistence across restarts.
    """

    def __init__(self, embed_fn, dimension: int = 768):
        import faiss
        import numpy as np
        self.embed_fn = embed_fn
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after norm)
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
        self._np = np

    def add(self, texts: list[str], metadatas: list[dict]) -> None:
        import numpy as np
        embs = np.array(self.embed_fn(texts), dtype="float32")
        faiss.normalize_L2(embs)
        self.index.add(embs)
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        import numpy as np
        if self.index.ntotal == 0:
            return []
        q = np.array(self.embed_fn([query]), dtype="float32")
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "text": self.documents[idx],
                "source": self.metadatas[idx].get("source", ""),
                "score": float(score),
                "metadata": self.metadatas[idx],
            })
        return results

    def count(self) -> int:
        return self.index.ntotal


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Vector store operations")
    parser.add_argument(
        "--action",
        choices=["ingest", "query", "stats", "reset"],
        required=True,
    )
    parser.add_argument("--query", type=str, default="diabetes mellitus treatment")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    settings.ensure_dirs()
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    if args.action == "ingest":
        embed_fn = get_embedding_function()
        ingest_all_processed(collection, embed_fn)

    elif args.action == "query":
        embed_fn = get_embedding_function()
        results = mmr_retrieval(args.query, collection, embed_fn, top_k=args.top_k)
        print(f"\nTop {args.top_k} results for: '{args.query}'\n{'='*60}")
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] Score: {r['score']:.4f} | Source: {r['source']}")
            print(f"    {r['text'][:200]}…")

    elif args.action == "stats":
        count = collection.count()
        print(f"\nCollection: {settings.chroma_collection_name}")
        print(f"Total documents: {count:,}")

    elif args.action == "reset":
        client.delete_collection(settings.chroma_collection_name)
        logger.warning("Collection deleted.")


if __name__ == "__main__":
    main()
