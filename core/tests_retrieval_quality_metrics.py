import unittest

from core.ml.evaluation import (
    candidate_recall_at_k,
    classify_answer_outcome,
    compute_metrics,
)


class CandidateRecallMetricTests(unittest.TestCase):
    def test_candidate_recall_is_independent_of_post_rerank_order(self):
        common = {
            "candidate_chunk_ids": [2, 3, 4],
            "relevant_chunk_ids": [2, 4],
        }
        first = compute_metrics([{
            **common,
            "ranked_chunk_ids": [2, 9],
        }])
        second = compute_metrics([{
            **common,
            "ranked_chunk_ids": [9, 4],
        }])

        self.assertEqual(first["candidate_recall@5"], 1.0)
        self.assertEqual(second["candidate_recall@5"], 1.0)
        self.assertNotEqual(first["hit@1"], second["hit@1"])
        self.assertNotEqual(first["mrr@10"], second["mrr@10"])

    def test_candidate_recall_supports_five_ten_and_twenty(self):
        candidates = list(range(100, 106)) + [7] + list(range(106, 114)) + [8]
        relevant = {7, 8}
        self.assertEqual(candidate_recall_at_k(candidates, relevant, 5), 0.0)
        self.assertEqual(candidate_recall_at_k(candidates, relevant, 10), 0.5)
        self.assertEqual(candidate_recall_at_k(candidates, relevant, 20), 1.0)

        metrics = compute_metrics([{
            "candidate_chunk_ids": candidates,
            "ranked_chunk_ids": [7, 8],
            "relevant_chunk_ids": relevant,
        }], k_values=(5, 10, 20))
        self.assertEqual(metrics["candidate_recall@5"], 0.0)
        self.assertEqual(metrics["candidate_recall@10"], 0.5)
        self.assertEqual(metrics["candidate_recall@20"], 1.0)
        self.assertEqual(metrics["candidate_recall_denominator"], 1)

    def test_candidate_recall_evaluated_flag_excludes_unmeasured_candidates(self):
        metrics = compute_metrics([{
            "candidate_chunk_ids": [],
            "candidate_recall_evaluated": False,
            "ranked_chunk_ids": [],
            "relevant_chunk_ids": [1],
        }])
        self.assertIsNone(metrics["candidate_recall@5"])
        self.assertEqual(metrics["candidate_recall_denominator"], 0)
        self.assertEqual(metrics["n_candidate_recall_not_evaluated"], 1)

    def test_empty_but_evaluated_candidate_list_counts_as_a_candidate_miss(self):
        metrics = compute_metrics([{
            "candidate_chunk_ids": [],
            "candidate_recall_evaluated": True,
            "ranked_chunk_ids": [],
            "relevant_chunk_ids": [1],
        }])
        self.assertEqual(metrics["candidate_recall@5"], 0.0)
        self.assertEqual(metrics["candidate_recall_denominator"], 1)
        self.assertEqual(metrics["n_candidate_recall_not_evaluated"], 0)

    def test_candidate_recall_requires_candidate_list(self):
        metrics = compute_metrics([{
            "ranked_chunk_ids": [1],
            "relevant_chunk_ids": [1],
        }])
        self.assertEqual(metrics["hit@1"], 1.0)
        self.assertIsNone(metrics["candidate_recall@5"])
        self.assertEqual(metrics["candidate_recall_denominator"], 0)
        self.assertEqual(metrics["n_candidate_recall_missing"], 1)


class RetrievalDenominatorTests(unittest.TestCase):
    def test_empty_evaluation_has_no_fabricated_zero_scores(self):
        metrics = compute_metrics([], k_values=(5, 10, 20))
        self.assertEqual(metrics["status"], "NO_EVALUATION_RECORDS")
        self.assertEqual(metrics["retrieval_denominator"], 0)
        self.assertEqual(metrics["candidate_recall_denominator"], 0)
        for k in (5, 10, 20):
            self.assertIsNone(metrics[f"hit@{k}"])
            self.assertIsNone(metrics[f"recall@{k}"])
            self.assertIsNone(metrics[f"precision@{k}"])
            self.assertIsNone(metrics[f"candidate_recall@{k}"])
        self.assertIsNone(metrics["mrr@10"])
        self.assertIsNone(metrics["ndcg@10"])

    def test_all_expected_abstentions_have_zero_retrieval_denominators(self):
        metrics = compute_metrics([{
            "candidate_chunk_ids": [1],
            "ranked_chunk_ids": [1],
            "relevant_chunk_ids": [1],
            "expected_abstain": True,
            "observed_abstain": True,
        }])
        self.assertEqual(metrics["status"], "NO_SUPPORTED_QUERIES")
        self.assertEqual(metrics["n_total_queries"], 1)
        self.assertEqual(metrics["n_supported_queries"], 0)
        self.assertEqual(metrics["n_unsupported_queries"], 1)
        self.assertEqual(metrics["retrieval_denominator"], 0)
        self.assertEqual(metrics["candidate_recall_denominator"], 0)
        self.assertIsNone(metrics["hit@1"])
        self.assertIsNone(metrics["candidate_recall@5"])
        self.assertIsNone(metrics["mrr@10"])
        self.assertEqual(metrics["abstention_denominator"], 1)
        self.assertEqual(metrics["n_correct_abstentions"], 1)
        self.assertEqual(metrics["abstention_accuracy"], 1.0)

    def test_expected_abstain_and_observed_abstain_drive_correctness(self):
        metrics = compute_metrics([
            {
                "ranked_chunk_ids": [1],
                "relevant_chunk_ids": [1],
                "expected_abstain": True,
                "observed_abstain": False,
                "is_unsupported": False,
            },
            {
                "ranked_chunk_ids": [],
                "relevant_chunk_ids": [2],
                "expected_abstain": False,
                "observed_abstain": True,
                "is_unsupported": True,
            },
        ])
        # Explicit expected labels take precedence over legacy is_unsupported.
        self.assertEqual(metrics["n_supported_queries"], 1)
        self.assertEqual(metrics["n_unsupported_queries"], 1)
        self.assertEqual(metrics["abstention_denominator"], 1)
        self.assertEqual(metrics["n_correct_abstentions"], 0)
        self.assertEqual(metrics["abstention_accuracy"], 0.0)
        self.assertEqual(metrics["n_false_abstentions"], 1)
        self.assertEqual(metrics["false_abstention_denominator"], 1)
        self.assertEqual(metrics["false_abstention_rate"], 1.0)
        self.assertEqual(metrics["abstention_decision_denominator"], 2)
        self.assertEqual(metrics["abstention_decision_correct"], 0)
        self.assertEqual(metrics["abstention_decision_accuracy"], 0.0)

    def test_legacy_is_unsupported_remains_supported(self):
        metrics = compute_metrics([{
            "ranked_chunk_ids": [],
            "relevant_chunk_ids": [1],
            "is_unsupported": True,
        }])
        self.assertEqual(metrics["n_unsupported_queries"], 1)
        self.assertEqual(metrics["retrieval_denominator"], 0)
        self.assertEqual(metrics["abstention_denominator"], 1)
        self.assertEqual(metrics["abstention_accuracy"], 1.0)

    def test_precision_keeps_requested_k_denominator_for_short_lists(self):
        metrics = compute_metrics([{
            "ranked_chunk_ids": [3, 7],
            "candidate_chunk_ids": [3, 7],
            "relevant_chunk_ids": [7],
        }], k_values=(1, 5, 10))
        self.assertEqual(metrics["precision@1"], 0.0)
        self.assertEqual(metrics["precision@5"], 0.2)
        self.assertEqual(metrics["precision@10"], 0.1)
        self.assertEqual(metrics["candidate_recall@5"], 1.0)


class AnswerOutcomeClassificationTests(unittest.TestCase):
    def test_insufficient_engine_classifies_custom_message_without_references(self):
        outcome = classify_answer_outcome({
            "engine": "insufficient-evidence",
            "answer": "The available records do not support a conclusion.",
            "references": [],
        })
        self.assertEqual(outcome, {
            "outcome": "INSUFFICIENT_EVIDENCE",
            "has_references": False,
            "observed_abstain": True,
        })

    def test_policy_guard_is_other_even_without_references(self):
        outcome = classify_answer_outcome({
            "engine": "policy-guard",
            "answer": "I cannot help with that request.",
            "references": [],
        })
        self.assertEqual(outcome, {
            "outcome": "OTHER",
            "has_references": False,
            "observed_abstain": False,
        })

    def test_insufficient_engine_with_references_is_not_a_correct_abstention(self):
        outcome = classify_answer_outcome({
            "engine": "insufficient-evidence",
            "answer": "No supported answer is available.",
            "references": [],
            "citations": [{"document_id": 1, "chunk_id": 2}],
        })
        self.assertEqual(outcome, {
            "outcome": "OTHER",
            "has_references": True,
            "observed_abstain": False,
        })

    def test_cited_answer_is_evidence_supported(self):
        outcome = classify_answer_outcome({
            "engine": "source-structured-extractive",
            "answer": "A cited answer.",
            "references": [{"document_id": 1, "chunk_id": 2}],
        })
        self.assertEqual(outcome["outcome"], "EVIDENCE_SUPPORTED")
        self.assertTrue(outcome["has_references"])
        self.assertFalse(outcome["observed_abstain"])


if __name__ == "__main__":
    unittest.main()
