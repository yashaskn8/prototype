"""Offline safety regressions for the three-asset SME training export."""
import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.servy_three_asset_training_export import (
    ASSETS, DOCUMENTS, sha256_file,
)


class ThreeAssetTrainingExportTests(unittest.TestCase):
    def test_export_is_exactly_nine_manual_bindings(self):
        self.assertEqual(set(ASSETS), {59, 60, 61})
        self.assertEqual(set(DOCUMENTS), set(range(168, 177)))
        for asset_id in ASSETS:
            rows = {
                document_id: doc_type
                for document_id, (bound_asset, doc_type) in DOCUMENTS.items()
                if bound_asset == asset_id
            }
            self.assertEqual(len(rows), 3)
            self.assertEqual(
                set(rows.values()),
                {"troubleshooting", "maintenance", "installation"},
            )

    def test_export_does_not_admit_any_document_without_a_reviewer(self):
        source = Path(__file__).with_name("servy_three_asset_training_export.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"review_status": "pending_sme_review"', source)
        self.assertIn('"training_status": "NOT_TRAINING_INPUT_PENDING_SME_REVIEW"', source)
        self.assertIn("positive_chunk_ids", source)
        self.assertIn("hard_negative_chunk_ids", source)
        self.assertIn("reviewer", source)
        self.assertIn("source_family", source)

    def test_sha256_helper_hashes_exact_attachment_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "approved-manual.pdf"
            path.write_bytes(b"synthetic-test-attachment-bytes")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(sha256_file(path), expected)


if __name__ == "__main__":
    unittest.main()
