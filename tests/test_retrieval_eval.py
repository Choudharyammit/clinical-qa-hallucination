"""
Unit tests for retrieval evaluation metrics.
"""

import pytest

from src.evaluation.retrieval_eval import (
    RetrievalMetrics,
    average_precision,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestRecallAtK:
    def test_perfect_recall(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_zero_recall(self):
        assert recall_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5

    def test_empty_relevant(self):
        assert recall_at_k(["a", "b"], set(), k=2) == 0.0

    def test_k_smaller_than_retrieved(self):
        # Only top-2 of ["a", "b", "c"] are considered
        assert recall_at_k(["a", "x", "b"], {"a", "b"}, k=2) == 0.5

    def test_k_larger_than_retrieved(self):
        assert recall_at_k(["a"], {"a", "b"}, k=10) == 0.5


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_second_position(self):
        assert abs(reciprocal_rank(["x", "a", "b"], {"a"}) - 0.5) < 1e-6

    def test_third_position(self):
        assert abs(reciprocal_rank(["x", "y", "a"], {"a"}) - 1/3) < 1e-6

    def test_not_found(self):
        assert reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0

    def test_empty_retrieved(self):
        assert reciprocal_rank([], {"a"}) == 0.0


class TestNDCGAtK:
    def test_perfect_ndcg(self):
        # All relevant docs at top → NDCG = 1.0
        assert abs(ndcg_at_k(["a", "b"], {"a", "b"}, k=2) - 1.0) < 1e-6

    def test_zero_ndcg(self):
        assert ndcg_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0

    def test_partial_ndcg_ordering_matters(self):
        # Relevant doc at position 1 > position 2
        ndcg_first = ndcg_at_k(["a", "x"], {"a"}, k=2)
        ndcg_second = ndcg_at_k(["x", "a"], {"a"}, k=2)
        assert ndcg_first > ndcg_second

    def test_empty_relevant(self):
        assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0

    def test_k_cutoff(self):
        # Relevant doc is at position 3 but k=2 → not counted
        assert ndcg_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0


class TestAveragePrecision:
    def test_perfect_ap(self):
        assert abs(average_precision(["a", "b"], {"a", "b"}) - 1.0) < 1e-6

    def test_zero_ap(self):
        assert average_precision(["x", "y"], {"a", "b"}) == 0.0

    def test_empty_relevant(self):
        assert average_precision(["a", "b"], set()) == 0.0

    def test_single_relevant_at_rank_1(self):
        assert abs(average_precision(["a", "x", "y"], {"a"}) - 1.0) < 1e-6

    def test_single_relevant_at_rank_2(self):
        ap = average_precision(["x", "a", "y"], {"a"})
        assert abs(ap - 0.5) < 1e-6


class TestHitRateAtK:
    def test_hit(self):
        assert hit_rate_at_k(["a", "b", "c"], {"a"}, k=3) == 1.0

    def test_miss(self):
        assert hit_rate_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0

    def test_k_cutoff(self):
        # Relevant at position 3 but k=2
        assert hit_rate_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0
        assert hit_rate_at_k(["x", "y", "a"], {"a"}, k=3) == 1.0


class TestRetrievalMetrics:
    def test_summary_has_all_keys(self):
        m = RetrievalMetrics(
            mrr=0.5,
            map_score=0.4,
            recall_at_k={5: 0.6, 10: 0.8},
            ndcg_at_k={5: 0.55, 10: 0.75},
            hit_rate_at_k={5: 0.7, 10: 0.9},
            n_queries=100,
        )
        s = m.summary()
        assert "mrr" in s
        assert "recall@5" in s
        assert "recall@10" in s
        assert "ndcg@5" in s
        assert "ndcg@10" in s
        assert "n_queries" in s

    def test_summary_rounds_values(self):
        m = RetrievalMetrics(
            mrr=0.123456789,
            recall_at_k={5: 0.66666},
        )
        s = m.summary()
        assert len(str(s["mrr"]).split(".")[-1]) <= 4
