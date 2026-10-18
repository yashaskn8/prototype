"""Unit tests for the one-shot authorized review sign-off workflow tool."""
import json
import tempfile
import unittest
from pathlib import Path

from servy_v4_one_shot_signoff import sign_off


class TestServyV4OneShotSignoff(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

        self.corpus_file = self.root / "corpus.jsonl"
        self.corpus_file.write_text(
            json.dumps({"chunk_id": 1, "tenant_id": "9", "document_id": "100", "passage": "Compressor instructions"}) + "\n" +
            json.dumps({"chunk_id": 2, "tenant_id": "9", "document_id": "100", "passage": "Safety instructions"}) + "\n",
            encoding="utf-8"
        )

        self.bundle_dir = self.root / "bundle"
        self.bundle_dir.mkdir()
        (self.bundle_dir / "train_unapproved_candidate.jsonl").write_text(
            json.dumps({
                "query_id": "q1", "query_text": "repair compressor", "tenant_id": "9",
                "positive_chunk_ids": [1], "negative_chunk_ids": [2]
            }) + "\n",
            encoding="utf-8"
        )
        (self.bundle_dir / "dev_unapproved_candidate.jsonl").write_text("", encoding="utf-8")

        self.pkg_dir = self.root / "package"
        self.pkg_dir.mkdir()
        (self.pkg_dir / "source_family_recommendations_and_evidence.jsonl").write_text(
            json.dumps({
                "document_id": "100", "tenant_id": "9", "document_title": "Compressor Manual",
                "product": "Compressor", "proposed_canonical_source_family": "FAMILY-COMPRESSOR",
                "proposed_source_version": "1.0", "chunk_count": 2,
                "source_document_sha256": ["0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"]
            }) + "\n",
            encoding="utf-8"
        )
        (self.pkg_dir / "question_passage_recommendations.jsonl").write_text(
            json.dumps({
                "split": "train", "query_id": "q1", "role": "positive", "chunk_id": "1",
                "document_id": "100", "heading": "## Repair"
            }) + "\n" +
            json.dumps({
                "split": "train", "query_id": "q1", "role": "negative", "chunk_id": "2",
                "document_id": "100", "heading": "## Safety"
            }) + "\n",
            encoding="utf-8"
        )

        self.out_dir = self.root / "out_adjudication"

    def test_missing_reviewer_id_fails(self):
        with self.assertRaises(ValueError):
            sign_off(self.pkg_dir, self.bundle_dir, self.corpus_file, self.out_dir,
                     reviewer_id="", approval_reference="AUTH-01",
                     confirm_high_risk_reviewed=True,
                     confirm_company_policy_permits=True)

    def test_missing_confirmation_fails(self):
        with self.assertRaises(ValueError):
            sign_off(self.pkg_dir, self.bundle_dir, self.corpus_file, self.out_dir,
                     reviewer_id="reviewer-1", approval_reference="AUTH-01",
                     confirm_high_risk_reviewed=False,
                     confirm_company_policy_permits=True)

    def test_successful_sign_off(self):
        res = sign_off(self.pkg_dir, self.bundle_dir, self.corpus_file, self.out_dir,
                       reviewer_id="lead-auditor-1", approval_reference="AUTH-REF-2026",
                       confirm_high_risk_reviewed=True,
                       confirm_company_policy_permits=True)
        self.assertEqual(res["status"], "SIGN_OFF_BUNDLE_PREPARED")
        self.assertTrue((self.out_dir / "source_family_review_queue.jsonl").exists())
        self.assertTrue((self.out_dir / "expert_review_queue.jsonl").exists())
        self.assertTrue((self.out_dir / "reviewer_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
