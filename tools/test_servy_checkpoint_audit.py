"""Synthetic CPU-only regression tests; no database or external model."""
import unittest

from servy_checkpoint_audit import (
    normalize_inputs, validate_candidates, check_split_overlap,
    rerank_fixed, evaluate_one,
)


class CheckpointAuditTests(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            {"chunk_id": 1, "tenant_id": 9, "document_id": 31, "passage": "pressure valve"},
            {"chunk_id": 2, "tenant_id": 9, "document_id": 32, "passage": "temperature alert"},
            {"chunk_id": 3, "tenant_id": 9, "document_id": 31, "passage": "valve details"},
            {"chunk_id": 4, "tenant_id": 8, "document_id": 99, "passage": "other tenant"},
        ]
        self.dev = [{"query_id": "q-1", "tenant_id": 9,
                     "query_text": "pressure valve", "positive_chunk_ids": [1, 3]}]
        self.cases, self.chunks = normalize_inputs(self.dev, self.corpus)

    def test_fixed_candidate_comparison(self):
        candidates = validate_candidates(self.cases, self.chunks,
            [{"query_id": "q-1", "chunk_ids": [2, 1, 3]}])
        before = evaluate_one(self.cases, self.chunks, candidates,
                              lambda pairs: [4.0, 1.0], pool_size=2)
        after = evaluate_one(self.cases, self.chunks, candidates,
                             lambda pairs: [-2.0, 9.0], pool_size=2)
        self.assertEqual(before["queries"][0]["gold_rank"], 2)
        self.assertEqual(after["queries"][0]["gold_rank"], 1)
        self.assertEqual(before["metrics"]["hit_at_1"], 0)
        self.assertEqual(after["metrics"]["hit_at_1"], 100)
        self.assertEqual(after["queries"][0]["ranked_chunk_ids"][-1], "3")

    def test_reject_cross_tenant_candidate(self):
        with self.assertRaisesRegex(ValueError, "cross-tenant"):
            validate_candidates(self.cases, self.chunks,
                                [{"query_id": "q-1", "chunk_ids": [4, 1]}])

    def test_reject_cross_tenant_gold(self):
        with self.assertRaisesRegex(ValueError, "different tenant"):
            normalize_inputs([{"query_id": "q", "tenant_id": 9,
                               "query_text": "x", "chunk_id": 4}], self.corpus)

    def test_reject_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "Duplicate corpus"):
            normalize_inputs(self.dev, self.corpus + self.corpus[:1])
        with self.assertRaisesRegex(ValueError, "Duplicate query"):
            normalize_inputs(self.dev * 2, self.corpus)

    def test_reject_missing_or_duplicate_candidates(self):
        with self.assertRaisesRegex(ValueError, "Candidate query IDs differ"):
            validate_candidates(self.cases, self.chunks, [])
        with self.assertRaisesRegex(ValueError, "duplicate candidate"):
            validate_candidates(self.cases, self.chunks,
                                [{"query_id": "q-1", "chunk_ids": [1, 1]}])

    def test_overlap_checks(self):
        collisions = check_split_overlap(self.cases, [
            {"query_text": " Pressure  Valve ", "positive_chunk_ids": [1]}
        ], self.chunks)
        self.assertEqual(collisions[0]["collision_types"],
                         ["query", "gold_chunk", "source_document"])

    def test_invalid_scores(self):
        with self.assertRaisesRegex(ValueError, "score count"):
            rerank_fixed("q", ["1", "2"], self.chunks, lambda pairs: [1])
        with self.assertRaisesRegex(ValueError, "Invalid reranker score"):
            rerank_fixed("q", ["1"], self.chunks,
                         lambda pairs: [float("nan")])

    def test_ties_and_untouched_tail(self):
        ranks, scores = rerank_fixed("q", ["2", "1", "3"],
                                      self.chunks, lambda pairs: [0, 0], 2)
        self.assertEqual(ranks, ["2", "1", "3"])
        self.assertEqual(scores, {"2": 0, "1": 0})


if __name__ == "__main__":
    unittest.main()
