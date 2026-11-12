"""Focused regression tests for serving evaluation instrumentation."""
from django.test import SimpleTestCase

from core.ml.evaluation import classify_answer_outcome
from core.ml.evaluate_benchmark import (
    _per_asset_metric_breakdown,
    _serving_metric_summary,
)


class ServingAnswerOutcomeTests(SimpleTestCase):
    def test_custom_insufficient_evidence_response_is_not_a_direct_answer(self):
        result = classify_answer_outcome({
            "engine": "insufficient-evidence",
            "answer": "I could not verify enough evidence for this request.",
            "references": [],
        })
        self.assertEqual(result["outcome"], "INSUFFICIENT_EVIDENCE")
        self.assertTrue(result["observed_abstain"])

    def test_policy_guard_is_other_not_an_evidence_abstention(self):
        result = classify_answer_outcome({
            "engine": "policy-guard",
            "answer": "I cannot help with that request.",
            "references": [],
        })
        self.assertEqual(result["outcome"], "OTHER")
        self.assertFalse(result["observed_abstain"])

    def test_citation_on_insufficient_response_is_not_a_correct_abstention(self):
        result = classify_answer_outcome({
            "engine": "insufficient-evidence",
            "answer": "I could not verify enough evidence.",
            "references": [{"chunk_id": 81}],
        })
        self.assertEqual(result["outcome"], "OTHER")
        self.assertFalse(result["observed_abstain"])

    def test_source_evidence_does_not_imply_procedure_approval(self):
        """Provenance-backed evidence and ProcedureApproval are separate gates."""
        result = classify_answer_outcome({
            "engine": "extractive",
            "answer": "The passage reports an observed condition.",
            "references": [{
                "chunk_id": 81,
                "source_verified": True,
                "customer_action_authorized": False,
            }],
        })
        self.assertEqual(result["outcome"], "EVIDENCE_SUPPORTED")
        self.assertNotIn("procedure_authorized", result)

    def test_abstention_correctness_uses_expected_response_not_empty_retrieval(self):
        metrics = _serving_metric_summary([
            {
                "query_id": "unsupported-1",
                "ranked_chunk_ids": [81],
                "candidate_chunk_ids": [81],
                "candidate_recall_evaluated": True,
                "relevant_chunk_ids": [],
                "is_unsupported": True,
                "expected_abstain": True,
                "observed_abstain": True,
                "answer_mode": "INSUFFICIENT_EVIDENCE",
                "answer_correctness_label": None,
            },
            {
                "query_id": "supported-1",
                "ranked_chunk_ids": [81],
                "candidate_chunk_ids": [81],
                "candidate_recall_evaluated": True,
                "relevant_chunk_ids": [81],
                "is_unsupported": False,
                "expected_abstain": False,
                "observed_abstain": True,
                "answer_mode": "INSUFFICIENT_EVIDENCE",
                "answer_correctness_label": None,
            },
        ])
        self.assertEqual(metrics["abstention_correctness"]["correct"], 1)
        self.assertEqual(metrics["abstention_correctness"]["denominator"], 2)
        self.assertEqual(metrics["abstention_correctness"]["accuracy"], 0.5)
        self.assertEqual(metrics["response_type_counts"]["INSUFFICIENT_EVIDENCE"], 2)
        self.assertEqual(metrics["answer_correctness"]["denominator"], 0)
        self.assertIsNone(metrics["answer_correctness"]["accuracy"])

    def test_answer_correctness_is_separate_and_requires_explicit_review_labels(self):
        metrics = _serving_metric_summary([
            {"expected_abstain": False, "observed_abstain": False,
             "answer_mode": "EVIDENCE_SUPPORTED", "answer_correctness_label": True},
            {"expected_abstain": False, "observed_abstain": False,
             "answer_mode": "OTHER", "answer_correctness_label": False},
        ])
        self.assertEqual(metrics["answer_correctness"]["correct"], 1)
        self.assertEqual(metrics["answer_correctness"]["denominator"], 2)
        self.assertEqual(metrics["answer_correctness"]["accuracy"], 0.5)
        self.assertEqual(metrics["abstention_correctness"]["accuracy"], 1.0)


class PerAssetMetricTests(SimpleTestCase):
    def test_assets_59_to_61_include_counts_and_denominators(self):
        records = {
            "59": [{
                "ranked_chunk_ids": [1], "candidate_chunk_ids": [1, 2, 3, 4, 5],
                "candidate_recall_evaluated": True, "relevant_chunk_ids": [1, 6],
                "is_unsupported": False, "expected_abstain": False,
                "observed_abstain": False, "answer_mode": "EVIDENCE_SUPPORTED",
                "answer_correctness_label": True,
            }],
            "60": [{
                "ranked_chunk_ids": [2], "candidate_chunk_ids": [2],
                "candidate_recall_evaluated": True, "relevant_chunk_ids": [9],
                "is_unsupported": False, "expected_abstain": False,
                "observed_abstain": False, "answer_mode": "OTHER",
                "answer_correctness_label": False,
            }],
        }
        result = _per_asset_metric_breakdown(records)
        self.assertEqual(set(result), {"59", "60", "61"})
        self.assertEqual(result["59"]["n_queries_total"], 1)
        self.assertEqual(result["59"]["retrieval_denominator"], 1)
        self.assertEqual(result["59"]["candidate_recall_denominator"], 1)
        self.assertEqual(result["59"]["candidate_recall@5"], 0.5)
        self.assertEqual(result["60"]["n_supported_queries"], 1)
        self.assertEqual(result["61"]["n_queries_total"], 0)
        self.assertEqual(result["61"]["retrieval_denominator"], 0)
        self.assertIsNone(result["61"]["hit@1"])
