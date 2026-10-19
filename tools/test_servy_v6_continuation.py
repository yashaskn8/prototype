"""Synthetic, model-free safety tests for Servy V6 continuation workflow."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from servy_v6_continuation import (
    V5_WEIGHTS_SHA256, check_permission, training_plan,
    verify_v5_checkpoint,
)


class V6ContinuationSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.v5 = self.root / "best_model"
        self.v5.mkdir()
        (self.v5 / "model.safetensors").write_bytes(b"synthetic-only")
        (self.v5 / "config.json").write_text("{}", encoding="utf-8")
        self.out = self.root / "v6_distinct"

    def test_v5_requires_exact_known_checkpoint_hash(self):
        with self.assertRaisesRegex(ValueError, "V5_CHECKPOINT_WEIGHTS_SHA256_MISMATCH"):
            verify_v5_checkpoint(self.v5)

    def test_v5_requires_model_files(self):
        (self.v5 / "config.json").unlink()
        with self.assertRaisesRegex(ValueError, "V5_CHECKPOINT_FILE_MISSING"):
            verify_v5_checkpoint(self.v5)

    def test_verified_v5_model_identity(self):
        with mock.patch("servy_v6_continuation.sha256", return_value=V5_WEIGHTS_SHA256):
            result = verify_v5_checkpoint(self.v5)
        self.assertEqual(result["weights_sha256"], V5_WEIGHTS_SHA256)

    def test_planning_is_not_permission_or_training(self):
        admitted = {
            "status": "READY_FOR_EXPLORATORY_FINE_TUNING",
            "candidate_train_sha256": "sha-selected",
            "original_train_sha256": "sha-original",
            "train_queries": 111,
            "positive_pairs": 111,
            "negative_pairs": 161,
            "dev_queries": 42,
            "quarantined_negative_references": 109,
        }
        with mock.patch("servy_v6_continuation.admit", return_value=admitted), \
             mock.patch("servy_v6_continuation.verify_v5_checkpoint",
                        return_value={"weights_sha256": V5_WEIGHTS_SHA256,
                                      "model_path": str(self.v5)}):
            result = training_plan(
                split_dir=self.root, corpus=self.root / "c",
                families=self.root / "f", candidates=self.root / "z",
                candidate_train=self.root / "subset.jsonl",
                manifest=self.root / "manifest.json",
                v5_model=self.v5, out_dir=self.out,
            )
        self.assertEqual(result["train_queries"], 111)
        self.assertTrue(result["starts_from_v5_weights"])
        self.assertTrue(result["uses_selected_subset_only"])
        self.assertFalse(result["training_performed"])
        self.assertFalse(result["external_training_authorized"])
        self.assertFalse(result["independent_holdout_verified"])
        self.assertFalse(self.out.exists())

    def test_plan_cannot_overwrite_v5(self):
        admitted = {
            "status": "READY_FOR_EXPLORATORY_FINE_TUNING",
            "candidate_train_sha256": "sha-selected",
            "original_train_sha256": "sha-original",
            "train_queries": 111,
            "positive_pairs": 111,
            "negative_pairs": 161,
            "dev_queries": 42,
            "quarantined_negative_references": 109,
        }
        with mock.patch("servy_v6_continuation.admit", return_value=admitted), \
             mock.patch("servy_v6_continuation.verify_v5_checkpoint",
                        return_value={"weights_sha256": V5_WEIGHTS_SHA256,
                                      "model_path": str(self.v5)}), \
             mock.patch("servy_v6_continuation.check_storage"):
            with self.assertRaisesRegex(ValueError, "OVERLAP_V5_CHECKPOINT"):
                training_plan(
                    split_dir=self.root, corpus="corpus", families="families",
                    candidates="candidates", candidate_train=self.root / "selected.jsonl",
                    manifest="manifest", v5_model=self.v5, out_dir=self.v5 / "nested",
                )

    def permission(self, overrides=None):
        record = {
            "status": "APPROVED_FOR_RESEARCH_TRAINING",
            "approved_by": "Company authorized reviewer",
            "approval_reference": "company-record-123",
            "selected_train_sha256": "selected-sha256",
            "environment": "local_secure",
            "external_processor_approved": False,
        }
        record.update(overrides or {})
        path = self.root / "permission.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    def test_missing_permission_is_blocked(self):
        with self.assertRaisesRegex(ValueError, "RECORD_REQUIRED"):
            check_permission(None, selected_sha256="selected-sha256",
                             environment="local_secure")

    def test_pending_signoff_is_blocked(self):
        file = self.permission({"status": "PENDING_AUTHORIZED_SIGN_OFF"})
        with self.assertRaisesRegex(ValueError, "NOT_APPROVED"):
            check_permission(file, selected_sha256="selected-sha256",
                             environment="local_secure")

    def test_missing_approver_identity_is_blocked(self):
        file = self.permission({"approved_by": ""})
        with self.assertRaisesRegex(ValueError, "REVIEWER_AND_APPROVAL"):
            check_permission(file, selected_sha256="selected-sha256",
                             environment="local_secure")

    def test_permission_cannot_cover_different_train_subset(self):
        file = self.permission()
        with self.assertRaisesRegex(ValueError, "FINGERPRINT_MISMATCH"):
            check_permission(file, selected_sha256="tampered-sha256",
                             environment="local_secure")

    def test_external_colab_requires_specific_company_permission(self):
        file = self.permission({"environment": "google_colab"})
        with self.assertRaisesRegex(ValueError, "EXTERNAL_PROCESSOR_NOT_APPROVED"):
            check_permission(file, selected_sha256="selected-sha256",
                             environment="google_colab")

    def test_local_permission_cannot_be_used_for_colab(self):
        file = self.permission()
        with self.assertRaisesRegex(ValueError, "PERMISSION_ENVIRONMENT_MISMATCH"):
            check_permission(file, selected_sha256="selected-sha256",
                             environment="google_colab")

    def test_explicit_environment_scoped_permission_record_is_structurally_readable(self):
        file = self.permission()
        checked = check_permission(
            file, selected_sha256="selected-sha256",
            environment="local_secure",
        )
        self.assertEqual(checked["permission_reference"], "company-record-123")


if __name__ == "__main__":
    unittest.main()
