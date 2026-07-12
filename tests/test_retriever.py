"""
Unit tests for RetrieverAgent and vectorstore utilities.
Uses mocking to avoid loading heavy models in CI.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agents.retriever import RetrieverAgent, RetrievedDocument


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_collection():
    collection = MagicMock()
    collection.count.return_value = 100
    collection.query.return_value = {
        "documents": [["Metformin is first-line for type 2 diabetes.", "Insulin therapy is used."]],
        "metadatas": [
            [{"source": "PubMed:12345", "pmid": "12345", "title": "Diabetes treatment"},
             {"source": "PubMed:67890", "pmid": "67890", "title": "Insulin study"}]
        ],
        "distances": [[0.1, 0.3]],
    }
    return collection


@pytest.fixture
def mock_embed_fn():
    def embed(texts):
        return [[0.1] * 768 for _ in texts]
    return embed


@pytest.fixture
def retriever(mock_collection, mock_embed_fn):
    agent = RetrieverAgent(top_k=5, use_mmr=False)
    agent._collection = mock_collection
    agent._embed_fn = mock_embed_fn
    return agent


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestRetrieverAgent:
    def test_retrieve_returns_list(self, retriever):
        docs = retriever.retrieve("What is the treatment for diabetes?")
        assert isinstance(docs, list)

    def test_retrieve_returns_retrieved_documents(self, retriever):
        docs = retriever.retrieve("diabetes treatment")
        for doc in docs:
            assert isinstance(doc, RetrievedDocument)

    def test_retrieve_assigns_ranks(self, retriever):
        docs = retriever.retrieve("diabetes")
        for i, doc in enumerate(docs, start=1):
            assert doc.rank == i

    def test_retrieve_empty_collection(self, mock_embed_fn):
        empty_collection = MagicMock()
        empty_collection.count.return_value = 0
        agent = RetrieverAgent()
        agent._collection = empty_collection
        agent._embed_fn = mock_embed_fn
        docs = agent.retrieve("diabetes")
        assert docs == []

    def test_retrieve_for_context_returns_tuple(self, retriever):
        context, docs = retriever.retrieve_for_context("diabetes")
        assert isinstance(context, str)
        assert isinstance(docs, list)

    def test_context_contains_source(self, retriever):
        context, _ = retriever.retrieve_for_context("diabetes")
        assert "Source:" in context

    def test_recall_at_k_perfect(self, retriever):
        recall = retriever.recall_at_k("diabetes", relevant_pmids=["12345"], k=5)
        assert recall == 1.0

    def test_recall_at_k_zero(self, retriever):
        recall = retriever.recall_at_k("diabetes", relevant_pmids=["99999"], k=5)
        assert recall == 0.0

    def test_recall_at_k_no_relevant(self, retriever):
        recall = retriever.recall_at_k("diabetes", relevant_pmids=[], k=5)
        assert recall == 0.0

    def test_mrr_with_relevant_doc(self, retriever):
        mrr = retriever.mean_reciprocal_rank("diabetes", relevant_pmids=["12345"], k=5)
        assert mrr > 0.0

    def test_mrr_no_relevant(self, retriever):
        mrr = retriever.mean_reciprocal_rank("diabetes", relevant_pmids=["99999"], k=5)
        assert mrr == 0.0


class TestRetrievedDocument:
    def test_to_dict(self):
        doc = RetrievedDocument(
            text="sample text",
            source="PubMed:1",
            score=0.85,
            metadata={"pmid": "1"},
            rank=1,
        )
        d = doc.to_dict()
        assert d["text"] == "sample text"
        assert d["score"] == 0.85
        assert d["rank"] == 1
