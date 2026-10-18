"""Synthetic review-finalization tests. Never use production data."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from servy_v4_finalize_review import finalize, ReviewBlocked
from servy_training_preflight import inspect_inputs
from servy_v4_colab_train import make_bce_pairs


class ReviewFinalizationTests(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            {"chunk_id": "1", "tenant_id": "9", "document_id": "10",
             "passage": "Compressor procedure", "source_version": "1", "source_checksum_sha256": "a"*64},
            {"chunk_id": "2", "tenant_id": "9", "document_id": "20",
             "passage": "Sensor calibration", "source_version": "1", "source_checksum_sha256": "b"*64},
            {"chunk_id": "3", "tenant_id": "9", "document_id": "30",
             "passage": "Unrelated checklist", "source_version": "1", "source_checksum_sha256": "c"*64},
            {"chunk_id": "4", "tenant_id": "9", "document_id": "40",
             "passage": "Unrelated valve", "source_version": "1", "source_checksum_sha256": "d"*64},
        ]
        self.train = [{"tenant_id": "9", "query_id": "T1", "query_text": "compressor fault",
                       "positive_chunk_ids": ["1"], "negative_chunk_ids": ["3"]}]
        self.dev = [{"tenant_id": "9", "query_id": "D1", "query_text": "calibrate sensor",
                     "positive_chunk_ids": ["2"], "negative_chunk_ids": ["4"]}]
        self.families = [{
            "tenant_id": "9", "document_id": str(i*10),
            "decision": "approved",
            "verified_canonical_source_family": "family-"+str(i),
            "approved_source_version": "1",
            "source_document_sha256": [chr(96+i)*64],
            "reviewer_id": "reviewer", "evidence_reference": "file:///review/"+str(i),
        } for i in range(1,5)]
        self.reviews = []
        for split, qid, ids in (("train", "T1", [("positive","1"),("negative","3")]),
                                 ("dev","D1",[("positive","2"),("negative","4")])):
            self.reviews.append({
                "split": split, "query_id": qid,
                "references": [{
                    "role": role, "chunk_id": cid,
                    "document_id": self.corpus[int(cid)-1]["document_id"],
                    "passage": self.corpus[int(cid)-1]["passage"],
                    "decision": "approved", "reviewer_id": "reviewer",
                    "review_evidence_reference": "file:///decisions/"+qid+cid
                } for role,cid in ids]
            })
        self.manifest = {"authorized_reviewer_ids": ["reviewer"],
                         "approval_reference": "external-supervisor-approval"}

    def apply(self):
        return finalize(self.train, self.dev, self.corpus, self.reviews,
                        self.families, self.manifest)

    def test_approved_synthetic_full_export(self):
        exports, corpus, report = self.apply()
        self.assertEqual(report["result"], "REVIEW_RECORDS_STRUCTURALLY_CONSISTENT")
        self.assertEqual(report["reviewed_pairs"], 4)
        self.assertEqual(exports["train"][0]["positive_texts"], ["Compressor procedure"])
        self.assertEqual(exports["train"][0]["negative_texts"], ["Unrelated checklist"])
        self.assertEqual(exports["dev"][0]["negative_review_status"], "approved")
        self.assertTrue(all(c["source_group_status"] == "VERIFIED" for c in corpus))
        self.assertTrue(report["training_not_performed"])

    def test_pending_label_is_not_approved(self):
        self.reviews[0]["references"][0]["decision"] = "PENDING"
        with self.assertRaisesRegex(ReviewBlocked, "Expert passage decision"):
            self.apply()

    def test_missing_negative_review_blocks(self):
        self.reviews[0]["references"].pop()
        with self.assertRaisesRegex(ReviewBlocked, "Every positive and negative"):
            self.apply()

    def test_changing_passage_after_review_blocks(self):
        self.corpus[0]["passage"] = "Changed by another source"
        with self.assertRaisesRegex(ReviewBlocked, "Passage text changed"):
            self.apply()

    def test_cross_tenant_reference_blocks(self):
        self.corpus[0]["tenant_id"] = "8"
        with self.assertRaisesRegex(ReviewBlocked, "Missing source document|cross-tenant"):
            self.apply()

    def test_missing_source_family_review_blocks(self):
        self.families.pop()
        with self.assertRaisesRegex(ReviewBlocked, "All source documents"):
            self.apply()

    def test_wrong_source_checksum_blocks(self):
        self.families[0]["source_document_sha256"] = ["0"*64]
        with self.assertRaisesRegex(ReviewBlocked, "Reviewed checksum"):
            self.apply()

    def test_identical_checksum_cannot_have_two_families(self):
        self.corpus[1]["source_checksum_sha256"] = "a"*64
        self.families[1]["source_document_sha256"] = ["a"*64]
        with self.assertRaisesRegex(ReviewBlocked, "Identical original source"):
            self.apply()

    def test_cross_split_family_collision_blocks(self):
        self.families[1]["verified_canonical_source_family"] = "family-1"
        with self.assertRaisesRegex(ReviewBlocked, "leak across train and dev"):
            self.apply()

    def test_unlisted_reviewer_blocks(self):
        self.reviews[0]["references"][0]["reviewer_id"] = "unknown"
        with self.assertRaisesRegex(ReviewBlocked, "Expert passage decision"):
            self.apply()

    def test_rejected_source_mapping_blocks(self):
        self.families[0]["decision"] = "rejected"
        with self.assertRaisesRegex(ReviewBlocked, "Unapproved or unattested"):
            self.apply()

    def test_structurally_reviewed_export_matches_real_v4_strict_gate(self):
        exports, corpus, _ = self.apply()
        self.assertEqual(len(make_bce_pairs(exports["train"])), 2)
        self.assertEqual(len(make_bce_pairs(exports["dev"])), 2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def write(name, rows):
                path = root / (name + ".jsonl")
                path.write_text("".join(json.dumps(r) + "\n" for r in rows),
                                encoding="utf-8")
                return path
            data = {
                "train": write("train", exports["train"]),
                "dev": write("dev", exports["dev"]),
                "corpus": write("corpus", corpus),
                "candidates": write("candidates", [
                    {"query_id": "D1", "chunk_ids": ["2", "4"]}
                ]),
            }
            model = root / "model.json"
            model.write_text(json.dumps({
                "base_model_id": "BAAI/bge-reranker-v2-m3",
                "base_revision": "a"*40,
                "tokenizer_id": "BAAI/bge-reranker-v2-m3",
                "tokenizer_revision": "a"*40,
                "sentence_transformers_version": "5.1.0",
                "loss_name": "BinaryCrossEntropyLoss",
                "max_seq_length": 256,
                "candidate_pool": 10,
                "query_format": "raw-question",
                "passage_format": "approved-frozen-corpus"
            }), encoding="utf-8")
            target = root / "never_created_checkpoints"
            authorization = root / "authorization.json"
            authorization.write_text(json.dumps({
                "authorized_input_sha256": {
                    role: hashlib.sha256(path.read_bytes()).hexdigest()
                    for role, path in data.items()
                },
                "authorized_model_config_sha256":
                    hashlib.sha256(model.read_bytes()).hexdigest(),
                "authorized_reviewer_ids": ["reviewer"],
                "approval_reference": "SYNTHETIC_TEST_ONLY",
                "approved_persistent_destination": str(target),
            }), encoding="utf-8")
            result = inspect_inputs(
                str(data["train"]), str(data["dev"]),
                str(data["corpus"]), str(model), str(authorization),
                str(target), candidates=str(data["candidates"])
            )
            self.assertEqual(result["status"], "READY_FOR_RESEARCH_EXPERIMENT",
                             result["issues"])
            self.assertFalse(result["training_performed"])


if __name__ == "__main__":
    unittest.main()
