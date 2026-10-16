"""Offline safety and review gates for experimental Operation 22 components."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from django.test import SimpleTestCase

from core.ml.op22_review_workflow import (
    ReviewValidationError, freeze_reviewed_dataset, validate_approved_case,
)
from core.ml.op22_technical_evidence import (
    assess_passage, parse_technical_query, rank_authorized_candidates, verify_answer_evidence,
)
from tools.op22_starlly_snapshot import paged_records


def doc(**overrides):
    fields = {"asset_id": None, "product_id": None, "category_id": None, "domain_id": None, "asset": None}
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TechnicalEvidenceTests(SimpleTestCase):
    def test_parser_preserves_fault_codes_models_units_and_negation(self):
        q = parse_technical_query("Model VRF-CS-1.5T alarm E-04, fan not spinning at 230 V; inspect capacitor")
        self.assertIn("E04", q.fault_codes)
        self.assertIn("VRF-CS-1.5T", q.models)
        self.assertIn("capacitor", q.components)
        self.assertIn("inspect", q.actions)
        self.assertIn("spinning", q.negated_terms)
        self.assertIn("230v", q.numeric_expressions)
        self.assertIn("VRF", q.abbreviations)
        self.assertGreaterEqual(len(q.conditions), 2)

    def test_generic_symptom_is_relevant_but_not_answer_supported(self):
        q = parse_technical_query("Abnormal Noise")
        result = assess_passage(q, heading="Symptom: Noise or Vibration", body="Record the noise location.", document=doc(), source_verified=True)
        self.assertEqual(result.decision, "PARTIALLY_SUPPORTED")
        self.assertFalse(verify_answer_evidence(q, [{"chunk_id": 1, "assessment": result}])["answer_coverage"])

    def test_all_missing_evidence_is_answer_not_in_kb(self):
        q = parse_technical_query("What torque specification applies?")
        result = assess_passage(q, heading="Overview", body="General equipment description.", document=doc(), source_verified=True)
        self.assertEqual(verify_answer_evidence(q, [{"chunk_id": 1, "assessment": result}])["decision"], "ANSWER_NOT_IN_KB")

    def test_exact_code_cannot_be_guessed_or_prefix_matched(self):
        q = parse_technical_query("What does alarm E-04 mean?")
        result = assess_passage(q, heading="E-040 fault", body="Check the controller.", document=doc(), source_verified=True)
        self.assertEqual(result.decision, "ANSWER_NOT_IN_KB")
        self.assertFalse(result.exact_identifier_match)

    def test_missing_torque_value_is_not_supported(self):
        q = parse_technical_query("What flare nut torque specification applies?")
        result = assess_passage(q, heading="Installation", body="Check the flare nut and refrigerant line.", document=doc(), source_verified=True)
        self.assertEqual(result.decision, "ANSWER_NOT_IN_KB")
        self.assertFalse(result.specification_present)

    def test_source_and_equipment_gates(self):
        q = parse_technical_query("Inspect fan motor")
        foreign = doc(asset_id=17)
        asset = SimpleNamespace(id=18, product=None, product_id=None)
        mismatch = assess_passage(q, heading="Fan motor", body="Inspect fan motor wiring.", document=foreign, asset=asset, source_verified=True)
        self.assertEqual(mismatch.decision, "EQUIPMENT_MISMATCH")
        unverified = assess_passage(q, heading="Fan motor", body="Inspect fan motor wiring.", document=doc(), source_verified=False)
        self.assertEqual(unverified.decision, "SOURCE_UNVERIFIED")
        default_unverified = assess_passage(q, heading="Fan motor", body="Inspect fan motor wiring.", document=doc())
        self.assertEqual(default_unverified.decision, "SOURCE_UNVERIFIED")

    def test_wrong_explicit_model_and_safe_spelling(self):
        q = parse_technical_query("Model VRF-CS-1.5T inspect capcitor")
        self.assertIn("capacitor", q.components)
        other_model = doc(asset=SimpleNamespace(model_number="VRF-CS-2T"))
        result = assess_passage(q, heading="Electrical", body="Inspect capacitor on VRF-CS-2T.",
                                document=other_model, source_verified=True)
        self.assertEqual(result.decision, "EQUIPMENT_MISMATCH")
        unspecified_model = assess_passage(q, heading="Electrical", body="Inspect capacitor.",
                                           document=doc(), source_verified=True)
        self.assertEqual(unspecified_model.decision, "EQUIPMENT_MISMATCH")

    def test_hazardous_procedure_requires_approval(self):
        q = parse_technical_query("How to discharge capacitor?")
        result = assess_passage(q, heading="Capacitor discharge", body="Step 1: Isolate power and discharge capacitor.", document=doc(), source_verified=True)
        self.assertEqual(result.decision, "SAFETY_APPROVAL_REQUIRED")
        self.assertFalse(verify_answer_evidence(q, [{"chunk_id": 1, "assessment": result}])["answer_coverage"])

    def test_exact_code_evidence_ranks_above_generic_overview(self):
        q = parse_technical_query("Inspect controller for E-04 fault")
        chunks = [
            SimpleNamespace(id=1, heading="General Overview", text="Generic equipment overview.", document=doc(), document_id=11),
            SimpleNamespace(id=2, heading="E-04 fault", text="Inspect controller and check cable damage.", document=doc(), document_id=11),
        ]
        ranked = rank_authorized_candidates(q, chunks, {1: 1, 2: 2}, {1: 2, 2: 1}, verified_document_ids={11})
        self.assertEqual(ranked[0]["chunk_id"], 2)
        self.assertEqual(ranked[0]["assessment"].decision, "ANSWER_SUPPORTED")


class ReviewWorkflowTests(SimpleTestCase):
    def approved(self):
        return {
            "query_id": "NEW-1", "question_text": "Documented question", "review_status": "APPROVED",
            "reviewer_identity": "authorized-technician", "reviewed_at": "2026-10-16T10:00:00Z",
            "approval_evidence": "signed-review-record-1", "source_organisation": "273", "tenant_id": 9,
            "source_sha256": "a" * 64, "source_document_id": 200, "source_attachment_id": 10,
            "source_verification": "VERIFIED_ORIGINAL_COMPANY_SOURCE",
            "answerability": "ANSWERABLE", "valid_positive_chunk_ids": [300],
        }

    def test_pending_review_and_unknown_reviewer_never_become_ground_truth(self):
        row = self.approved()
        row["review_status"] = "PENDING_REVIEW"
        with self.assertRaises(ReviewValidationError):
            validate_approved_case(row, {"authorized-technician"})
        row["review_status"] = "APPROVED"
        with self.assertRaises(ReviewValidationError):
            validate_approved_case(row, set())

    def test_unanswerable_cannot_keep_positive_and_test_must_be_source_separated(self):
        row = self.approved()
        row["answerability"] = "UNANSWERABLE_FROM_APPROVED_SOURCES"
        with self.assertRaises(ReviewValidationError):
            validate_approved_case(row, {"authorized-technician"})
        row["answerability"] = "ANSWERABLE"
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "reviewed-v1.jsonl"
            with self.assertRaises(ReviewValidationError):
                freeze_reviewed_dataset([row], target, authorized_reviewers={"authorized-technician"}, excluded_source_hashes={"a" * 64})
            record = freeze_reviewed_dataset([row], target, authorized_reviewers={"authorized-technician"}, excluded_source_hashes=set())
            self.assertEqual(record["case_count"], 1)
            with self.assertRaises(ReviewValidationError):
                freeze_reviewed_dataset([row], target, authorized_reviewers={"authorized-technician"}, excluded_source_hashes=set())

    def test_alternative_positive_cannot_be_a_training_negative(self):
        row = self.approved()
        row["valid_alternative_chunk_ids"] = [301]
        with self.assertRaises(ReviewValidationError):
            validate_approved_case(row, {"authorized-technician"})
        row["valid_positive_chunk_ids"] = [300, 301]
        row["reviewed_negative_chunk_ids"] = [301]
        with self.assertRaises(ReviewValidationError):
            validate_approved_case(row, {"authorized-technician"})


class StarllySnapshotTests(SimpleTestCase):
    def test_complete_pages_and_duplicate_rejection(self):
        client = Mock()
        client._request_get.side_effect = [
            SimpleNamespace(json=lambda: [{"id": 1, "count": 3}, {"id": 2}]),
            SimpleNamespace(json=lambda: [{"id": 3, "count": 3}]),
        ]
        self.assertEqual(len(paged_records(client, "assets", page_size=2)), 3)
        client._request_get.side_effect = [
            SimpleNamespace(json=lambda: [{"id": 1, "count": 3}, {"id": 2}]),
            SimpleNamespace(json=lambda: [{"id": 1, "count": 3}]),
        ]
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            paged_records(client, "assets", page_size=2)
