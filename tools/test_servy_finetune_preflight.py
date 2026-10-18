"""Synthetic offline tests for Servy's no-training admission validator."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from servy_finetune_preflight import audit


class TestTrainingAdmission(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.base = Path(self.tempdir.name)
        self.corpus = [
            {"chunk_id": 1, "tenant_id": 9, "document_id": 10, "passage": "compressor fault alpha"},
            {"chunk_id": 2, "tenant_id": 9, "document_id": 20, "passage": "temperature check beta"},
            {"chunk_id": 3, "tenant_id": 9, "document_id": 30, "passage": "unrelated filter gamma"},
            {"chunk_id": 4, "tenant_id": 8, "document_id": 40, "passage": "another tenant delta"},
        ]
        self.train = [{"query_id": "T-1", "query_text": "why compressor fault",
                       "tenant_id": 9, "chunk_id": 1,
                       "positive_text": "compressor fault alpha"}]
        self.dev = [{"query_id": "D-1", "query_text": "temperature check",
                     "tenant_id": 9, "chunk_id": 2,
                     "positive_text": "temperature check beta"}]

    def dump(self, name, data):
        path = self.base / name
        with path.open("w", encoding="utf-8") as file:
            for row in data:
                file.write(json.dumps(row) + "\n")
        return str(path)

    def check(self, train=None, dev=None, corpus=None, stage2=None):
        a = self.dump("train.jsonl", self.train if train is None else train)
        b = self.dump("dev.jsonl", self.dev if dev is None else dev)
        c = self.dump("corpus.jsonl", self.corpus if corpus is None else corpus)
        d = self.dump("stage2.jsonl", stage2) if stage2 is not None else None
        return audit(a, b, c, d)

    def test_clean_research_case_never_claims_expert_approval(self):
        report = self.check()
        self.assertEqual(report["status"], "REVIEW_REQUIRED")
        self.assertEqual(report["numbers"]["blockers"], 0)
        self.assertTrue(report["training_not_performed"])
        self.assertFalse(report["fully_independent_test_verified"])

    def test_exact_positive_document_leak_fails(self):
        leaked_dev = [dict(self.dev[0], chunk_id=1, positive_text="compressor fault alpha")]
        report = self.check(dev=leaked_dev)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("SPLIT_DOCUMENT_COLLISION", report["issue_counts"])

    def test_negative_document_from_dev_fails(self):
        train = [dict(self.train[0], negative_chunk_ids=[2],
                      negative_text="temperature check beta")]
        report = self.check(train=train)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("SPLIT_DOCUMENT_COLLISION", report["issue_counts"])

    def test_same_positive_and_negative_fails(self):
        train = [dict(self.train[0], negative_chunk_ids=[1],
                      negative_text="compressor fault alpha")]
        report = self.check(train=train)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("CONTRADICTORY_POSITIVE_AND_NEGATIVE", report["issue_counts"])

    def test_missing_or_unmapped_source_fails(self):
        train = [dict(self.train[0], chunk_id=999, positive_text="nonexistent")]
        report = self.check(train=train)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("UNKNOWN_CHUNK_ID", report["issue_counts"])

    def test_cross_tenant_chunk_fails(self):
        train = [dict(self.train[0], chunk_id=4, positive_text="another tenant delta")]
        report = self.check(train=train)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("CROSS_TENANT_SOURCE", report["issue_counts"])

    def test_inconsistent_declared_document_fails(self):
        train = [dict(self.train[0], document_id=99)]
        report = self.check(train=train)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("DECLARED_DOCUMENT_MISMATCH", report["issue_counts"])

    def test_stage2_negative_from_dev_document_fails(self):
        stage2 = [
            {"query_id": "S-1", "query": "why compressor fault", "tenant_id": 9,
             "positive": "compressor fault alpha",
             "candidate_negative": "temperature check beta"}
        ]
        report = self.check(stage2=stage2)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("SPLIT_DOCUMENT_COLLISION", report["issue_counts"])

    def test_stage2_question_outside_first_train_split_fails(self):
        stage2 = [
            {"query_id": "S-1", "query": "a question never in training",
             "tenant_id": 9, "positive": "compressor fault alpha",
             "candidate_negative": "unrelated filter gamma"}
        ]
        report = self.check(stage2=stage2)
        self.assertIn("STAGE2_QUERY_NOT_IN_TRAIN_SPLIT", report["issue_counts"])

    def test_repeated_question_with_alternative_positive_warns(self):
        train = self.train + [
            {"query_id": "T-2", "query": "why compressor fault", "tenant_id": 9,
             "chunk_id": 3, "positive_text": "unrelated filter gamma"}
        ]
        report = self.check(train=train)
        self.assertIn("MULTI_POSITIVE_SAME_QUESTION_IN_BATCH", report["issue_counts"])

    def test_duplicate_corpus_ids_block(self):
        report = self.check(corpus=self.corpus + [self.corpus[0]])
        self.assertIn("DUPLICATE_CHUNK_ID", report["issue_counts"])

    def test_false_input_cannot_imply_zero_loss(self):
        report = self.check()
        self.assertNotIn("training_loss", report)
        self.assertEqual(report["status"], "REVIEW_REQUIRED")


    def test_mismatched_passage_even_with_correct_id_blocks(self):
        train = [dict(self.train[0], positive_text="different undocumented information")]
        report = self.check(train=train)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("PASSAGE_TEXT_DIFFERS_FROM_CORPUS", report["issue_counts"])

    def test_holdout_with_unanswerable_rows_does_not_require_gold(self):
        train_path = self.dump("train.jsonl", self.train)
        dev_path = self.dump("dev.jsonl", self.dev)
        corpus_path = self.dump("corpus.jsonl", self.corpus)
        holdout_path = self.dump("holdout.jsonl", [
            {"query_id": "H-1", "query_text": "no approved answer", "tenant_id": 9,
             "positive_chunk_ids": [], "answerability": "UNANSWERABLE_FROM_APPROVED_SOURCES"}
        ])
        report = audit(train_path, dev_path, corpus_path, holdout_path=holdout_path)
        self.assertEqual(report["numbers"]["blockers"], 0)

    def test_ephemeral_colab_checkpoint_is_blocked(self):
        train_path = self.dump("train.jsonl", self.train)
        dev_path = self.dump("dev.jsonl", self.dev)
        corpus_path = self.dump("corpus.jsonl", self.corpus)
        report = audit(train_path, dev_path, corpus_path,
                       artifact_dir="/content/checkpoints")
        self.assertIn("CHECKPOINT_TARGET_IS_EPHEMERAL_COLAB_STORAGE",
                      report["issue_counts"])
        self.assertEqual(report["status"], "BLOCKED")

    def test_planned_persistent_checkpoint_no_ephemeral_blocker(self):
        train_path = self.dump("train.jsonl", self.train)
        dev_path = self.dump("dev.jsonl", self.dev)
        corpus_path = self.dump("corpus.jsonl", self.corpus)
        report = audit(train_path, dev_path, corpus_path,
                       artifact_dir="C:/approved-model-storage")
        self.assertNotIn("CHECKPOINT_TARGET_IS_EPHEMERAL_COLAB_STORAGE",
                         report["issue_counts"])

    def test_same_document_negative_flags_unverified_semantics(self):
        corpus = self.corpus + [
            {"chunk_id": 5, "tenant_id": 9, "document_id": 10,
             "passage": "compressor nearby section epsilon"}
        ]
        train = [dict(self.train[0], negative_chunk_ids=[5],
                      negative_text="compressor nearby section epsilon")]
        report = self.check(train=train, corpus=corpus)
        self.assertIn("HARD_NEGATIVE_SAME_SOURCE_DOCUMENT", report["issue_counts"])


    def test_missing_training_passage_blocks_even_with_id(self):
        train = [dict(self.train[0], positive_text="")]
        report = self.check(train=train)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("MISSING_EXPORTED_TRAINING_PASSAGE", report["issue_counts"])

    def test_long_corpus_flags_tokenization_review(self):
        corpus = list(self.corpus)
        corpus[2] = dict(corpus[2], passage="a" * 1500)
        report = self.check(corpus=corpus)
        self.assertIn("CHECK_TOKENIZER_TRUNCATION_FOR_LONG_PASSAGES",
                      report["issue_counts"])


if __name__ == "__main__":
    unittest.main()
