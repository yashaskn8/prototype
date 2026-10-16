"""Synthetic checks for private retrieval; no source data or model downloads."""
import copy

from django.test import SimpleTestCase

from core.ml.op23_private_retrieval import PrivateRetrievalError, predict_minilm_passages


class FakeEncoder:
    def encode(self, texts, **kwargs):
        return [[1.0, 0.0] if "desired" in text.lower() else [0.0, 1.0] for text in texts]


class PrivateRetrievalTests(SimpleTestCase):
    def setUp(self):
        self.admission = {"tenant_id": 9, "sources": [{
            "source_id": "s1", "tenant_id": 9, "external_asset_id": 332,
            "source_verification": "VERIFIED_ORIGINAL_COMPANY_SOURCE",
            "attachment_name": "Manual", "asset_code": "A-332",
            "passages": [{"passage_id": "p1", "text": "Unrelated overview."},
                         {"passage_id": "p2", "text": "The desired definition."}],
        }]}
        self.case = {"query_id": "q1", "question_text": "desired", "answerability": "ANSWERABLE",
                     "external_asset_id": 332,
                     "searched_source_universe": {"source_ids": ["s1"], "indexed_source_ids": ["s1"],
                                                  "search_complete": True}}

    def test_answer_bearing_passage_is_ranked_without_generating_an_answer(self):
        prediction = predict_minilm_passages([self.case], self.admission, FakeEncoder())[0]
        self.assertEqual(prediction["ranked_passage_ids"], ["p2", "p1"])
        self.assertEqual(prediction["selected_source_id"], "s1")
        self.assertFalse(prediction["answered"])
        self.assertEqual(prediction["retrieved_tenant_ids"], [9, 9])

    def test_clarification_and_wrong_asset_are_safe(self):
        clarification = {"query_id": "q2", "answerability": "NEEDS_ASSET_CLARIFICATION",
                         "external_asset_id": None}
        prediction = predict_minilm_passages([clarification], self.admission, FakeEncoder())[0]
        self.assertEqual(prediction["decision"], "EQUIPMENT_CLARIFICATION_REQUIRED")
        wrong = copy.deepcopy(self.case)
        wrong["external_asset_id"] = 999
        with self.assertRaises(PrivateRetrievalError):
            predict_minilm_passages([wrong], self.admission, FakeEncoder())

    def test_incomplete_index_is_rejected(self):
        case = copy.deepcopy(self.case)
        case["searched_source_universe"]["search_complete"] = False
        with self.assertRaises(PrivateRetrievalError):
            predict_minilm_passages([case], self.admission, FakeEncoder())
