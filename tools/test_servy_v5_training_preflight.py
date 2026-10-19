"""Synthetic, offline self-tests for Servy V5 research training admission."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from servy_v5_training_preflight import admit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl(path, rows):
    Path(path).write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )


class ServyV5TrainingPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.split = self.root / "split"
        self.split.mkdir()
        self.original = [{
            "tenant_id": 1, "query_id": "t1",
            "query_text": "Check inlet pressure",
            "positive_chunk_ids": ["1"],
            "negative_chunk_ids": ["2"],
        }]
        write_jsonl(self.split / "train_unapproved_candidate.jsonl", self.original)
        self.train = self.root / "selected.jsonl"
        write_jsonl(self.train, self.original)
        self.corpus = self.root / "corpus.jsonl"
        write_jsonl(self.corpus, [
            {"chunk_id": "1", "tenant_id": 1, "passage": "Check inlet pressure"},
            {"chunk_id": "2", "tenant_id": 1, "passage": "Unrelated manual"},
        ])
        self.families = self.root / "families.jsonl"
        self.families.write_text("", encoding="utf-8")
        self.candidates = self.root / "candidates.jsonl"
        self.candidates.write_text("", encoding="utf-8")
        self.manifest = self.root / "manifest.json"
        self.save_manifest()

    def save_manifest(self):
        self.manifest.write_text(json.dumps({
            "expert_approved": False,
            "production_promotion_allowed": False,
            "accepted_training_queries": 1,
            "original_train_sha256": sha(self.split / "train_unapproved_candidate.jsonl"),
            "v5_candidate_sha256": sha(self.train),
        }), encoding="utf-8")

    def check(self):
        return admit(
            split_dir=self.split,
            corpus=self.corpus,
            families=self.families,
            candidates=self.candidates,
            candidate_train=self.train,
            manifest=self.manifest,
        )

    def mock_integrity(self):
        return mock.patch(
            "servy_v5_training_preflight.audited_inputs",
            return_value={
                "source_isolation_verified_against_proposed_families": True,
                "dev_queries": 42,
                "quarantined_negative_references": 109,
            },
        )

    def test_verified_subset_is_ready_for_research_only(self):
        with self.mock_integrity():
            report = self.check()
        self.assertEqual(report["status"], "READY_FOR_EXPLORATORY_FINE_TUNING")
        self.assertFalse(report["expert_labels_approved"])
        self.assertFalse(report["independent_holdout_verified"])
        self.assertFalse(report["production_promotion_allowed"])
        self.assertEqual(report["positive_pairs"], 1)
        self.assertEqual(report["negative_pairs"], 1)

    def test_candidate_rewrite_rejected_even_with_matching_manifest_hash(self):
        changed = [{**self.original[0], "query_text": "Unapproved rewrite"}]
        write_jsonl(self.train, changed)
        self.save_manifest()
        with self.mock_integrity(), self.assertRaisesRegex(ValueError, "MODIFIED"):
            self.check()

    def test_candidate_checksum_change_rejected(self):
        write_jsonl(self.train, [{**self.original[0], "query_text": "Tampered"}])
        with self.mock_integrity(), self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.check()

    def test_single_selected_query_without_negatives_is_rejected(self):
        self.original[0]["negative_chunk_ids"] = []
        write_jsonl(self.split / "train_unapproved_candidate.jsonl", self.original)
        write_jsonl(self.train, self.original)
        self.save_manifest()
        with self.mock_integrity(), self.assertRaisesRegex(ValueError, "EACH_SELECTED_QUERY"):
            self.check()

    def test_fabricated_expert_approval_rejected(self):
        report = json.loads(self.manifest.read_text(encoding="utf-8"))
        report["expert_approved"] = True
        self.manifest.write_text(json.dumps(report), encoding="utf-8")
        with self.mock_integrity(), self.assertRaisesRegex(ValueError, "expert approval"):
            self.check()


if __name__ == "__main__":
    unittest.main()
