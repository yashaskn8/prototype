"""Synthetic, CPU-only tests for the strict admission gate."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from servy_training_preflight import inspect_inputs


class TestStrictV4Admission(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train = [
            {"query_id": "q1", "query": "repair compressor", "tenant_id": 9,
             "positive_chunk_id": 1, "positive_text": "compressor repair procedure",
             "review_status": "approved", "reviewer_id": "domain-engineer-a"}
        ]
        self.dev = [
            {"query_id": "q2", "query": "calibrate sensor", "tenant_id": 9,
             "positive_chunk_id": 2, "positive_text": "sensor calibration sequence",
             "review_status": "approved", "reviewer_id": "domain-engineer-a"}
        ]
        self.corpus = [
            {"chunk_id": 1, "tenant_id": 9, "document_id": 100,
             "source_group": "manual-one", "source_version": "1",
             "passage": "compressor repair procedure"},
            {"chunk_id": 2, "tenant_id": 9, "document_id": 200,
             "source_group": "manual-two", "source_version": "1",
             "passage": "sensor calibration sequence"},
            {"chunk_id": 3, "tenant_id": 9, "document_id": 300,
             "source_group": "manual-three", "source_version": "1",
             "passage": "unrelated approved procedure"},
        ]
        self.model_config = {
            "base_model_id": "BAAI/bge-reranker-v2-m3",
            "base_revision": "pinned-revision-set-after-runtime-verification",
            "tokenizer_id": "BAAI/bge-reranker-v2-m3",
            "tokenizer_revision": "pinned-revision-set-after-runtime-verification",
            "sentence_transformers_version": "pin-after-runtime-verification",
            "loss_name": "BinaryCrossEntropyLoss",
            "max_seq_length": 256, "candidate_pool": 10,
            "query_format": "raw-question", "passage_format": "heading-body"
        }
        self.inputs = {}
        for key in ("train", "dev", "corpus"):
            path = self.root / (key + ".jsonl")
            with path.open("w", encoding="utf-8") as f:
                for row in getattr(self, key):
                    f.write(json.dumps(row) + "\n")
            self.inputs[key] = path
        self.model_path = self.root / "model.json"
        self.model_path.write_text(json.dumps(self.model_config), encoding="utf-8")
        self.destination = self.root / "approved-model-output"
        self.approval_path = self.root / "approval.json"
        self.refresh_approval()

    def refresh_approval(self):
        approved = {
            "authorized_input_sha256": {
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in self.inputs.items()
            },
            "authorized_reviewer_ids": ["domain-engineer-a"],
            "approved_persistent_destination": str(self.destination),
            "approval_reference": "synthetic-unit-test-only"
        }
        self.approval_path.write_text(json.dumps(approved), encoding="utf-8")

    def evaluate(self):
        return inspect_inputs(
            str(self.inputs["train"]), str(self.inputs["dev"]),
            str(self.inputs["corpus"]), str(self.model_path),
            str(self.approval_path), str(self.destination))

    def test_structurally_eligible_synthetic_research_does_not_prove_truth(self):
        report = self.evaluate()
        self.assertEqual(report["status"], "READY_FOR_RESEARCH_EXPERIMENT")
        self.assertFalse(report["training_performed"])
        self.assertTrue(report["limitations"])

    def test_unapproved_dataset_hash_is_rejected(self):
        with self.inputs["train"].open("a", encoding="utf-8") as f:
            f.write(" \n")
        self.assertEqual(self.evaluate()["status"], "BLOCKED_LABEL_VERIFICATION")

    def test_unauthorized_negative_reviewer_rejected(self):
        self.train[0].update(
            negative_chunk_id=3, negative_text="unrelated approved procedure",
            negative_review_status="approved", negative_reviewer_id="not-on-list"
        )
        self.inputs["train"].write_text(json.dumps(self.train[0]) + "\n", encoding="utf-8")
        self.refresh_approval()
        result = self.evaluate()
        self.assertEqual(result["status"], "BLOCKED_LABEL_VERIFICATION")
        self.assertIn("NEGATIVE_REVIEW_UNATTESTED", [i["code"] for i in result["issues"]])

    def test_source_family_collision_blocks(self):
        self.corpus[1]["source_group"] = "manual-one"
        self.inputs["corpus"].write_text(
            "".join(json.dumps(r) + "\n" for r in self.corpus), encoding="utf-8")
        self.refresh_approval()
        result = self.evaluate()
        self.assertEqual(result["status"], "BLOCKED_SPLIT_LEAKAGE")

    def test_missing_source_version_blocks(self):
        del self.corpus[0]["source_version"]
        self.inputs["corpus"].write_text(
            "".join(json.dumps(r) + "\n" for r in self.corpus), encoding="utf-8")
        self.refresh_approval()
        result = self.evaluate()
        self.assertEqual(result["status"], "BLOCKED_DATA_INTEGRITY")

    def test_existing_checkpoint_target_blocks(self):
        self.destination.mkdir()
        self.assertEqual(self.evaluate()["status"], "BLOCKED_MISSING_ARTIFACT")

    def test_missing_model_configuration_blocks(self):
        self.model_path.unlink()
        self.assertEqual(self.evaluate()["status"], "BLOCKED_MISSING_ARTIFACT")

    def test_mismatched_model_config_blocks(self):
        self.model_config.pop("tokenizer_revision")
        self.model_path.write_text(json.dumps(self.model_config), encoding="utf-8")
        self.assertEqual(self.evaluate()["status"], "BLOCKED_MODEL_COMPATIBILITY")


if __name__ == "__main__":
    unittest.main()
