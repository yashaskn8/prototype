"""Model-free tests for the Colab runner; no GPU, download or proprietary data."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from servy_v4_colab_train import (
    make_bce_pairs, raw_negative_texts, candidate_gate, check_backup_destination,
    validate_model_tokenizer_binding
)


class TestV4ColabSafety(unittest.TestCase):
    def test_preserves_technical_values_and_fault_codes(self):
        row = {
            "query": "Explain ERR-402",
            "positive_text": "Verify interlock input 24V before reset",
            "review_status": "approved",
            "negative_review_status": "approved",
            "negative_text": "ERR-402 applies to model ZX-41 at 24V DC",
        }
        records = make_bce_pairs([row])
        self.assertEqual(len(records), 2)
        self.assertEqual([r["label"] for r in records], [1.0, 0.0])
        self.assertIn("ERR-402", records[1]["text2"])
        self.assertIn("24V DC", records[1]["text2"])

    def test_missing_negative_review_fails(self):
        row = {
            "query": "What is the procedure", "positive_text": "Step 1",
            "negative_text": "Unrelated Step 2", "review_status": "approved",
        }
        with self.assertRaisesRegex(ValueError, "not been approved"):
            make_bce_pairs([row])

    def test_one_class_validation_is_invalid(self):
        row = {"query": "Q", "positive_text": "A", "review_status": "approved"}
        with self.assertRaisesRegex(ValueError, "both approved"):
            make_bce_pairs([row])

    def test_equivalent_positive_and_negative_blocks(self):
        row = {
            "query": "Q", "positive_text": "ERR-402 24V DC", "review_status": "approved",
            "negative_text": "err-402  24v dc", "negative_review_status": "approved"
        }
        with self.assertRaisesRegex(ValueError, "repeats"):
            make_bce_pairs([row])

    def test_model_tokenizer_binding_is_exact(self):
        valid = {
            "base_model_id": "BAAI/bge-reranker-v2-m3",
            "tokenizer_id": "BAAI/bge-reranker-v2-m3",
            "base_revision": "a" * 40, "tokenizer_revision": "a" * 40,
        }
        self.assertIsNone(validate_model_tokenizer_binding(valid))
        with self.assertRaisesRegex(ValueError, "tokenizer revision"):
            validate_model_tokenizer_binding(dict(valid, tokenizer_revision="b" * 40))
        with self.assertRaisesRegex(ValueError, "tokenizer ID"):
            validate_model_tokenizer_binding(dict(valid, tokenizer_id="other-model"))
        with self.assertRaisesRegex(ValueError, "reviewed BGE"):
            validate_model_tokenizer_binding(dict(valid, base_model_id="unauthorized-model"))

    def test_wrong_negative_list_schema_fails(self):
        with self.assertRaises(ValueError):
            raw_negative_texts({"negative_texts": "not-an-array"})

    def test_duplicate_labeled_pairs_fail(self):
        row = {
            "query": "Q", "positive_text": "A", "negative_text": "B",
            "review_status": "approved", "negative_review_status": "approved"
        }
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            make_bce_pairs([row, row])

    def test_frozen_candidates_bound_to_digest_and_tenant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def dump(name, rows):
                path = root / name
                path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
                return path
            corpus = dump("corpus.jsonl", [
                {"chunk_id": "1", "tenant_id": "9", "document_id": "10", "passage": "Correct"},
                {"chunk_id": "2", "tenant_id": "8", "document_id": "20", "passage": "Other"},
            ])
            dev = dump("dev.jsonl", [{
                "query_id": "q1", "query": "Correct question", "tenant_id": "9",
                "positive_chunk_id": "1"
            }])
            candidates = dump("candidates.jsonl", [
                {"query_id": "q1", "chunk_ids": ["1"]}
            ])
            approved = {"authorized_input_sha256": {
                "candidates": hashlib.sha256(candidates.read_bytes()).hexdigest()
            }}
            cases, chunks, frozen = candidate_gate(dev, corpus, candidates, approved)
            self.assertEqual(frozen["q1"], ["1"])
            approved["authorized_input_sha256"]["candidates"] = "tampered"
            with self.assertRaisesRegex(ValueError, "digest"):
                candidate_gate(dev, corpus, candidates, approved)
            approved["authorized_input_sha256"]["candidates"] = hashlib.sha256(candidates.read_bytes()).hexdigest()
            candidates.write_text(json.dumps({"query_id": "q1", "chunk_ids": ["2"]}) + "\n")
            approved["authorized_input_sha256"]["candidates"] = hashlib.sha256(candidates.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "cross-tenant"):
                candidate_gate(dev, corpus, candidates, approved)

    def test_refuses_existing_backup_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileExistsError):
                check_backup_destination(root)


if __name__ == "__main__":
    unittest.main()
