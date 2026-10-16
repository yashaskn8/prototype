"""Source admission and externally signed review regressions; synthetic fixtures only."""
import base64
import copy
import json
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from django.test import SimpleTestCase

from core.ml.op23_review_authority import ApprovalError, signed_content, verify_configuration_lock
from core.ml.op23_review_dataset import DatasetValidationError, freeze_cases, validate_case
from core.ml.op23_source_admission import (
    SourceAdmissionError, admit_capture, canonical_bytes, prior_source_record, sha256,
)


class AdmissionAndReviewTests(SimpleTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.capture = self.root / "capture"
        self.capture.mkdir()
        self.raw = b"new authentic manual bytes"
        self.text = "FAULT E-04: controller communication interruption.\n\nInspect the controller cable before reset."
        self.old_text = "Old belt manual: check belt tension at 12 mm.\n\nInspect guard before use."
        self.asset = {"id": 332, "org_id": 273, "code": "ASSET-332", "model_number": "MX-1",
                      "attachments": [{"id": 99, "name": "Controller service manual v1", "created_at": "2026-10-16"}]}
        self.complaint = {"id": 44, "asset": 332, "description": "What does E-04 mean?"}
        old_hash = sha256(b"old source bytes")
        self.baseline = {"excluded_raw_source_sha256": [old_hash], "historical_complaint_ids": [39, 40, 41, 42, 43],
                         "sources": [prior_source_record(raw_sha256=old_hash, text=self.old_text,
                                                         asset_code="ASSET-332", title="Belt manual v1")]}
        self.write_capture()
        self.admission = admit_capture(self.capture, self.baseline, expected_org="273", tenant_id=9)
        self.private_key = Ed25519PrivateKey.generate()
        public = self.private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        self.trust = {"company-key-test": {"active": True, "role": "technical_reviewer",
                                            "subject": "test-reviewer", "organisation": "273", "tenant_id": 9,
                                            "public_key_b64": base64.b64encode(public).decode()}}

    def write_capture(self):
        (self.capture / "original.bin").write_bytes(self.raw)
        (self.capture / "extracted.txt").write_text(self.text, encoding="utf-8")
        records = {
            "assets": [self.asset], "complaints": [self.complaint], "site_assets": [{"id": 1, "asset": 332}],
            "attachments": [{"external_asset_id": 332, "external_attachment_id": 99,
                             "name": "Controller service manual v1", "created_at": "2026-10-16",
                             "download_status": "DOWNLOADED_AND_EXTRACTED", "raw_sha256": sha256(self.raw),
                             "text_sha256": sha256(self.text.encode()), "raw_file": "original.bin",
                             "text_file": "extracted.txt"}],
        }
        for name, rows in records.items():
            (self.capture / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")
        manifest = {"catalog_authenticated": True, "organisation_id": "273", "tenant_id": 9}
        for name in records:
            manifest[f"{name}_sha256"] = sha256((self.capture / f"{name}.json").read_bytes())
        (self.capture / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def signed(self, **changes):
        source = self.admission["sources"][0]
        case = {"query_id": "NEW-44", "question_text": self.complaint["description"],
                "dataset_role": "INDEPENDENT_FINAL_TEST", "review_status": "APPROVED",
                "reviewer_identity": "test-reviewer", "reviewed_at": "2026-10-16T00:00:00Z",
                "source_organisation": "273", "tenant_id": 9,
                "query_provenance": {"kind": "starlly_complaint", "complaint_id": 44,
                                     "complaint_sha256": sha256(canonical_bytes(self.complaint))},
                "external_asset_id": 332, "evaluation_task": "TECHNICAL_ANSWERABILITY",
                "answerability": "ANSWERABLE", "source_id": source["source_id"],
                "source_sha256": source["raw_sha256"], "source_version": source["source_version"],
                "source_attachment_id": source["external_attachment_id"],
                "valid_positive_passage_ids": [source["passages"][0]["passage_id"]],
                "valid_alternative_passage_ids": [], "reviewed_negative_passage_ids": [],
                "searched_source_universe": {"source_ids": [source["source_id"]],
                                             "indexed_source_ids": [source["source_id"]],
                                             "search_complete": True}}
        case.update(changes)
        case["approval"] = {"key_id": "company-key-test",
                            "signature_b64": base64.b64encode(self.private_key.sign(signed_content(case))).decode()}
        return case

    def test_new_authorized_source_can_be_admitted_and_frozen(self):
        source = self.admission["sources"][0]
        self.assertEqual(source["independence_status"], "INDEPENDENT_SOURCE_CANDIDATE")
        case = self.signed()
        result = freeze_cases([case], self.root / "final.jsonl", admission=self.admission,
                              baseline=self.baseline, complaints=[self.complaint], trusted_keys=self.trust,
                              dataset_role="INDEPENDENT_FINAL_TEST")
        self.assertEqual(result["technical_case_count"], 1)

    def test_historical_and_near_duplicate_source_rejected(self):
        self.raw = b"old source bytes"
        self.write_capture()
        historical = admit_capture(self.capture, self.baseline, expected_org="273", tenant_id=9)
        self.assertEqual(historical["sources"][0]["independence_status"], "HISTORICAL_EXCLUDED")
        self.text = self.old_text
        self.raw = b"new bytes with old technical text"
        self.write_capture()
        near = admit_capture(self.capture, self.baseline, expected_org="273", tenant_id=9)
        self.assertEqual(near["sources"][0]["independence_status"], "NEAR_DUPLICATE_QUARANTINED")

    def test_text_only_historical_document_still_blocks_independence(self):
        baseline = copy.deepcopy(self.baseline)
        baseline["sources"].append(prior_source_record(raw_sha256="", text=self.text,
                                                        asset_code="UNSCOPED", title="Old imported note"))
        admitted = admit_capture(self.capture, baseline, expected_org="273", tenant_id=9)
        self.assertEqual(admitted["sources"][0]["independence_status"], "NEAR_DUPLICATE_QUARANTINED")

    def test_wrong_tenant_attachment_and_tampering_rejected(self):
        with self.assertRaises(SourceAdmissionError):
            admit_capture(self.capture, self.baseline, expected_org="273", tenant_id=8)
        self.asset["attachments"] = []
        self.write_capture()
        with self.assertRaises(SourceAdmissionError):
            admit_capture(self.capture, self.baseline, expected_org="273", tenant_id=9)
        self.asset["attachments"] = [{"id": 99, "name": "Controller service manual v1"}]
        self.write_capture()
        (self.capture / "original.bin").write_bytes(b"tampered")
        with self.assertRaises(SourceAdmissionError):
            admit_capture(self.capture, self.baseline, expected_org="273", tenant_id=9)

    def test_catalogue_attachment_metadata_mismatch_rejected(self):
        self.asset["attachments"][0]["name"] = "Different manual"
        self.write_capture()
        with self.assertRaises(SourceAdmissionError):
            admit_capture(self.capture, self.baseline, expected_org="273", tenant_id=9)

    def test_cross_organisation_complaint_rejected(self):
        self.complaint["org"] = 274
        self.write_capture()
        with self.assertRaises(SourceAdmissionError):
            admit_capture(self.capture, self.baseline, expected_org="273", tenant_id=9)

    def test_fake_reviewer_or_changed_document_fails(self):
        case = self.signed()
        forged = copy.deepcopy(case)
        forged["reviewer_identity"] = "attacker"
        with self.assertRaises(DatasetValidationError):
            validate_case(forged, admission=self.admission, baseline=self.baseline,
                          complaints=[self.complaint], trusted_keys=self.trust,
                          dataset_role="INDEPENDENT_FINAL_TEST")
        changed = copy.deepcopy(case)
        changed["source_version"] = "2026-10-17"
        with self.assertRaises(DatasetValidationError):
            validate_case(changed, admission=self.admission, baseline=self.baseline,
                          complaints=[self.complaint], trusted_keys=self.trust,
                          dataset_role="INDEPENDENT_FINAL_TEST")
        with self.assertRaises(DatasetValidationError):
            validate_case(case, admission=self.admission, baseline=self.baseline,
                          complaints=[self.complaint], trusted_keys={},
                          dataset_role="INDEPENDENT_FINAL_TEST")
        invalid_key = copy.deepcopy(self.trust)
        invalid_key["company-key-test"]["public_key_b64"] = "%%%"
        with self.assertRaises(DatasetValidationError):
            validate_case(case, admission=self.admission, baseline=self.baseline,
                          complaints=[self.complaint], trusted_keys=invalid_key,
                          dataset_role="INDEPENDENT_FINAL_TEST")

    def test_unanswerable_no_positive_or_source_and_clarification(self):
        unanswerable = self.signed(answerability="UNANSWERABLE_FROM_APPROVED_SOURCES",
                                   valid_positive_passage_ids=[], source_id=None, source_sha256=None,
                                   source_version=None, source_attachment_id=None)
        self.assertTrue(validate_case(unanswerable, admission=self.admission, baseline=self.baseline,
                                      complaints=[self.complaint], trusted_keys=self.trust,
                                      dataset_role="INDEPENDENT_FINAL_TEST"))
        invalid_negative = self.signed(answerability="UNANSWERABLE_FROM_APPROVED_SOURCES",
                                       valid_positive_passage_ids=[], source_id=None, source_sha256=None,
                                       source_version=None, source_attachment_id=None,
                                       reviewed_negative_passage_ids=["invented-passage"])
        with self.assertRaises(DatasetValidationError):
            validate_case(invalid_negative, admission=self.admission, baseline=self.baseline,
                          complaints=[self.complaint], trusted_keys=self.trust,
                          dataset_role="INDEPENDENT_FINAL_TEST")
        incomplete = self.signed(answerability="UNANSWERABLE_FROM_APPROVED_SOURCES",
                                 valid_positive_passage_ids=[], source_id=None, source_sha256=None,
                                 searched_source_universe={"source_ids": [], "indexed_source_ids": [], "search_complete": False})
        with self.assertRaises(DatasetValidationError):
            validate_case(incomplete, admission=self.admission, baseline=self.baseline,
                          complaints=[self.complaint], trusted_keys=self.trust,
                          dataset_role="INDEPENDENT_FINAL_TEST")
        no_asset = {"id": 45, "asset": None, "description": "Device not working"}
        clarification = self.signed(query_id="NEW-45", question_text=no_asset["description"],
                                    query_provenance={"kind": "starlly_complaint", "complaint_id": 45,
                                                      "complaint_sha256": sha256(canonical_bytes(no_asset))},
                                    external_asset_id=None, evaluation_task="CLARIFICATION",
                                    answerability="NEEDS_ASSET_CLARIFICATION", valid_positive_passage_ids=[],
                                    source_id=None, source_sha256=None, source_version=None,
                                    source_attachment_id=None)
        self.assertTrue(validate_case(clarification, admission=self.admission, baseline=self.baseline,
                                      complaints=[no_asset], trusted_keys=self.trust,
                                      dataset_role="INDEPENDENT_FINAL_TEST"))

    def test_historical_complaint_and_cross_role_rejected(self):
        case = self.signed()
        old = copy.deepcopy(self.baseline)
        old["historical_complaint_ids"].append(44)
        with self.assertRaises(DatasetValidationError):
            validate_case(case, admission=self.admission, baseline=old,
                          complaints=[self.complaint], trusted_keys=self.trust,
                          dataset_role="INDEPENDENT_FINAL_TEST")
        previous = copy.deepcopy(case)
        previous["dataset_role"] = "TRAIN"
        with self.assertRaises(DatasetValidationError):
            freeze_cases([case], self.root / "final2.jsonl", admission=self.admission,
                         baseline=self.baseline, complaints=[self.complaint], trusted_keys=self.trust,
                         dataset_role="INDEPENDENT_FINAL_TEST", existing_role_cases=[previous])

    def test_final_configuration_requires_separate_signed_owner_lock(self):
        owner_key = Ed25519PrivateKey.generate()
        owner_public = owner_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        trust = copy.deepcopy(self.trust)
        trust["owner-key-test"] = {"active": True, "role": "evaluation_owner", "subject": "test-owner",
                                   "organisation": "273", "tenant_id": 9,
                                   "public_key_b64": base64.b64encode(owner_public).decode()}
        lock = {"status": "FROZEN", "configuration_sha256": "abc", "source_organisation": "273",
                "tenant_id": 9, "owner_identity": "test-owner", "frozen_at": "2026-10-16T00:00:00Z"}
        lock["approval"] = {"key_id": "owner-key-test",
                            "signature_b64": base64.b64encode(owner_key.sign(signed_content(lock))).decode()}
        verify_configuration_lock(lock, trust, expected_org="273", tenant_id=9, configuration_sha256="abc")
        with self.assertRaises(ApprovalError):
            verify_configuration_lock(lock, trust, expected_org="273", tenant_id=9, configuration_sha256="changed")
        forged = copy.deepcopy(lock)
        forged["owner_identity"] = "other"
        with self.assertRaises(ApprovalError):
            verify_configuration_lock(forged, trust, expected_org="273", tenant_id=9, configuration_sha256="abc")
