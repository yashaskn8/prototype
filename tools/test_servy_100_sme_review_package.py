"""Synthetic checks for human-review separation and metric denominators."""
import unittest

from tools.servy_100_sme_review_package import _metrics


class ReviewMetricTests(unittest.TestCase):
    def case(self, outcome="ANSWERED"):
        return {"id": "SYNTH-1", "question_sha256": "synthetic-hash",
                "baseline": {"classification": "REFUSED"},
                "final": {"classification": outcome},
                "automated_checks": {"claim_evidence": [
                    {"claim_id": "c1", "evidence_id": "s1", "document_id": 1, "chunk_id": 2}
                ] if outcome == "ANSWERED" else []}}

    def decision(self):
        return {"id": "SYNTH-1", "question_sha256": "synthetic-hash",
                "review_status": "APPROVED", "reviewer_name": "Authorized synthetic reviewer",
                "reviewer_authorization": "synthetic test", "reviewed_at": "2026-10-31",
                "evidence_ids_consulted": ["s1"], "expected_handling": "ANSWERABLE",
                "critical_safety_or_authorization_failure": "NO",
                "final_factual_correctness": "CORRECT", "final_safety": "SAFE",
                "four_points": {str(n): "CORRECT" for n in range(1, 5)},
                "citation_links": [{"claim_id": "c1", "evidence_id": "s1",
                                    "document_id": 1, "chunk_id": 2,
                                    "semantic_support": "SUPPORTED"}]}

    def test_pending_decision_does_not_create_accuracy_claim(self):
        result = _metrics([self.case()], [{"id": "SYNTH-1", "review_status": "PENDING_REVIEW"}])
        self.assertEqual(result["reviewed_cases"], 0)
        self.assertEqual(result["complete_answer_accuracy"]["denominator"], 0)
        self.assertIsNone(result["complete_answer_accuracy"]["rate"])
        self.assertIsNone(result["semantic_citation_precision"]["rate"])

    def test_approved_review_uses_actual_denominators(self):
        result = _metrics([self.case()], [self.decision()])
        self.assertEqual(result["complete_answer_accuracy"],
                         {"correct": 1, "denominator": 1, "rate": 1.0})
        self.assertEqual(result["semantic_citation_precision"],
                         {"supported": 1, "denominator": 1, "rate": 1.0})

    def test_mismatched_citation_identity_is_rejected(self):
        decision = self.decision()
        decision["citation_links"][0]["chunk_id"] = 99
        with self.assertRaises(RuntimeError):
            _metrics([self.case()], [decision])

    def test_unreviewed_evidence_cannot_be_approved(self):
        decision = self.decision()
        decision["citation_links"][0]["semantic_support"] = "PENDING_REVIEW"
        with self.assertRaises(RuntimeError):
            _metrics([self.case()], [decision])


if __name__ == "__main__":
    unittest.main()
