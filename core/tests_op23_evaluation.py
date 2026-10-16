from django.test import SimpleTestCase

from core.ml.op23_evaluation import EvaluationError, evaluate_reviewed


class ReviewedEvaluationTests(SimpleTestCase):
    def test_multiple_equivalent_passages_and_roles_remain_separate(self):
        cases = [
            {"query_id": "dev-1", "dataset_role": "DEVELOPMENT", "review_status": "APPROVED",
             "tenant_id": 9, "answerability": "ANSWERABLE", "external_asset_id": 332,
             "source_id": "source-a", "valid_positive_passage_ids": ["passage-a", "passage-b"]},
            {"query_id": "final-1", "dataset_role": "INDEPENDENT_FINAL_TEST", "review_status": "APPROVED",
             "tenant_id": 9, "answerability": "UNANSWERABLE_FROM_APPROVED_SOURCES"},
        ]
        predictions = [
            {"query_id": "dev-1", "tenant_id": 9, "external_asset_id": 332,
             "selected_source_id": "source-a", "ranked_passage_ids": ["passage-b", "wrong"],
             "candidate_passage_ids": ["passage-a", "passage-b"],
             "selected_passage_id": "passage-b", "answered": True,
             "retrieved_tenant_ids": [9], "end_to_end_latency_ms": 12.5},
            {"query_id": "final-1", "tenant_id": 9, "answered": False,
             "retrieved_tenant_ids": [], "end_to_end_latency_ms": 10.2},
        ]
        report = evaluate_reviewed(cases, predictions)
        self.assertEqual(report["DEVELOPMENT"]["passage_hit_at_1"], 1)
        self.assertEqual(report["DEVELOPMENT"]["answer_bearing_evidence_rate_among_answered"], 1)
        self.assertEqual(report["DEVELOPMENT"]["answer_present_in_candidate_set"], 1)
        self.assertEqual(report["DEVELOPMENT"]["cross_tenant_leakage_rate"], 0)
        self.assertEqual(report["DEVELOPMENT"]["p95_end_to_end_latency_ms"], 12.5)
        self.assertIsNone(report["DEVELOPMENT"]["grounded_technical_answer_correctness"])
        self.assertIsNone(report["DEVELOPMENT"]["unsafe_action_rate"])
        self.assertEqual(report["INDEPENDENT_FINAL_TEST"]["unanswerable_abstention_rate"], 1)

    def test_pending_or_cross_tenant_prediction_rejected(self):
        row = {"query_id": "x", "dataset_role": "DEVELOPMENT", "review_status": "PENDING_REVIEW",
               "tenant_id": 9, "answerability": "ANSWERABLE", "valid_positive_passage_ids": ["a"]}
        with self.assertRaises(EvaluationError):
            evaluate_reviewed([row], [{"query_id": "x", "tenant_id": 9}])
        row["review_status"] = "APPROVED"
        with self.assertRaises(EvaluationError):
            evaluate_reviewed([row], [{"query_id": "x", "tenant_id": 8}])
