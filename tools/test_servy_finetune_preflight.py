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

    def check(self, train=None, dev=None, corpus=None, stage2=None, loss_name=None):
        a = self.dump("train.jsonl", self.train if train is None else train)
        b = self.dump("dev.jsonl", self.dev if dev is None else dev)
        c = self.dump("corpus.jsonl", self.corpus if corpus is None else corpus)
        d = self.dump("stage2.jsonl", stage2) if stage2 is not None else None
        return audit(a, b, c, d, loss_name=loss_name)

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


    def test_corpus_source_family_collision_without_row_group_blocks(self):
        corpus = [
            dict(self.corpus[0], source_group="manual-family-same"),
            dict(self.corpus[1], source_group="manual-family-same"),
            *self.corpus[2:],
        ]
        report = self.check(corpus=corpus)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("SPLIT_SOURCE_GROUP_COLLISION", report["issue_counts"])

    def test_in_batch_negative_of_other_positive_blocks(self):
        corpus = self.corpus + [
            {"chunk_id": 5, "tenant_id": 9, "document_id": 50,
             "passage": "new technician procedure"},
        ]
        train = [dict(self.train[0], negative_chunk_ids=[3],
                      negative_text="unrelated filter gamma"),
                 {"query_id": "T-2", "query": "filter repair", "tenant_id": 9,
                  "chunk_id": 3, "positive_text": "unrelated filter gamma"}]
        report = self.check(train=train, corpus=corpus)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("TRAIN_NEGATIVE_IS_ANOTHER_POSITIVE", report["issue_counts"])

    def test_equivalent_text_under_distinct_chunks_blocks(self):
        corpus = self.corpus + [
            {"chunk_id": 5, "tenant_id": 9, "document_id": 50,
             "passage": "compressor fault alpha"}
        ]
        train = [dict(self.train[0], negative_chunk_ids=[5],
                      negative_text="compressor fault alpha")]
        report = self.check(train=train, corpus=corpus)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("TRAIN_NEGATIVE_EQUIVALENT_POSITIVE_TEXT", report["issue_counts"])

    def test_stage2_negative_colliding_with_train_positive_blocks(self):
        corpus = self.corpus + [
            {"chunk_id": 5, "tenant_id": 9, "document_id": 50,
             "passage": "new technician procedure"}
        ]
        stage2 = [
            {"query_id": "S-1", "query": "why compressor fault", "tenant_id": 9,
             "positive": "compressor fault alpha",
             "candidate_negative": "compressor fault alpha"}
        ]
        report = self.check(corpus=corpus, stage2=stage2)
        self.assertIn("TRAIN_NEGATIVE_IS_ANOTHER_POSITIVE", report["issue_counts"])

    def test_holdout_missing_gold_not_silently_unanswerable(self):
        train = self.dump("train.jsonl", self.train)
        dev = self.dump("dev.jsonl", self.dev)
        corpus = self.dump("corpus.jsonl", self.corpus)
        holdout = self.dump("holdout.jsonl", [
            {"query_id": "H-1", "query": "needs a manual", "tenant_id": 9,
             "positive_chunk_ids": []}
        ])
        report = audit(train, dev, corpus, holdout_path=holdout)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("HOLDOUT_NO_GOLD_WITHOUT_UNANSWERABLE_LABEL", report["issue_counts"])


    def test_copied_passage_cross_split_alias_negative_blocks(self):
        # Same exact manual passage with a different source identity.
        # Old ID/document-only checks missed train-positive vs dev-negative.
        corpus = self.corpus + [
            {"chunk_id": 5, "tenant_id": 9, "document_id": 50,
             "source_group": "independent-looking-version",
             "passage": "compressor fault alpha"}
        ]
        dev = [dict(self.dev[0], negative_chunk_ids=[5],
                    negative_text="compressor fault alpha")]
        report = self.check(dev=dev, corpus=corpus)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("SPLIT_PASSAGE_CONTENT_COLLISION", report["issue_counts"])

    def test_copied_negative_passage_cross_split_alias_blocks(self):
        corpus = self.corpus + [
            {"chunk_id": 5, "tenant_id": 9, "document_id": 50,
             "source_group": "different-source",
             "passage": "unrelated filter gamma"}
        ]
        train = [dict(self.train[0], negative_chunk_ids=[3],
                      negative_text="unrelated filter gamma")]
        dev = [dict(self.dev[0], negative_chunk_ids=[5],
                    negative_text="unrelated filter gamma")]
        report = self.check(train=train, dev=dev, corpus=corpus)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("SPLIT_PASSAGE_CONTENT_COLLISION", report["issue_counts"])


    def test_ambiguous_text_only_gold_source_is_blocked(self):
        corpus = self.corpus + [
            {"chunk_id": 5, "tenant_id": 9, "document_id": 50,
             "passage": "compressor fault alpha"}
        ]
        train = [dict(self.train[0])]
        train[0].pop("chunk_id")
        report = self.check(train=train, corpus=corpus)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("AMBIGUOUS_IDENTICAL_PASSAGE", report["issue_counts"])

    def test_multiple_negative_ids_have_matching_exported_texts(self):
        corpus = self.corpus + [
            {"chunk_id": 5, "tenant_id": 9, "document_id": 50,
             "passage": "correctly declared negative"},
            {"chunk_id": 6, "tenant_id": 9, "document_id": 60,
             "passage": "wrong alternative negative"}
        ]
        train = [dict(self.train[0], negative_chunk_ids=[3, 5],
                      negative_texts=["unrelated filter gamma",
                                      "wrong alternative negative"])]
        report = self.check(train=train, corpus=corpus)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("NEGATIVE_ID_TEXT_SET_MISMATCH", report["issue_counts"])

    def test_bce_cross_question_negative_reuse_is_review_not_blocker(self):
        train = [
            dict(self.train[0], negative_chunk_ids=[3],
                 negative_text="unrelated filter gamma"),
            {"query_id": "T-2", "query_text": "filter repair procedure",
             "tenant_id": 9, "chunk_id": 3,
             "positive_text": "unrelated filter gamma"},
        ]
        result = self.check(train=train, loss_name="BinaryCrossEntropyLoss")
        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertEqual(result["numbers"]["blockers"], 0)
        self.assertIn("BCE_CROSS_QUERY_NEGATIVE_REUSE_REQUIRES_EXPERT_REVIEW",
                      result["issue_counts"])
        # Unknown or in-batch objectives retain the conservative safeguard.
        contrastive = self.check(train=train)
        self.assertIn("TRAIN_NEGATIVE_IS_ANOTHER_POSITIVE",
                      contrastive["issue_counts"])

    def test_bce_same_query_alternative_positive_negative_blocks(self):
        train = [
            dict(self.train[0], negative_chunk_ids=[3],
                 negative_text="unrelated filter gamma"),
            {"query_id": "T-2", "query_text": "why compressor fault",
             "tenant_id": 9, "chunk_id": 3,
             "positive_text": "unrelated filter gamma"},
        ]
        report = self.check(train=train, loss_name="BinaryCrossEntropyLoss")
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("BCE_NEGATIVE_IS_SAME_QUERY_POSITIVE",
                      report["issue_counts"])

    def test_conflicting_canonical_groups_within_document_block(self):
        corpus = self.corpus + [
            {"chunk_id": 5, "tenant_id": 9, "document_id": 10,
             "source_group": "manual-family-two",
             "passage": "different paragraph of compressor guide"}
        ]
        corpus[0] = dict(corpus[0], source_group="manual-family-one")
        report = self.check(corpus=corpus)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("CONFLICTING_SOURCE_GROUP_FOR_DOCUMENT",
                      report["issue_counts"])

    def test_conflicting_source_versions_within_document_block(self):
        corpus = self.corpus + [
            {"chunk_id": 5, "tenant_id": 9, "document_id": 10,
             "source_version": "v2",
             "passage": "updated document paragraph"}
        ]
        corpus[0] = dict(corpus[0], source_version="v1")
        report = self.check(corpus=corpus)
        self.assertIn("CONFLICTING_SOURCE_VERSION_FOR_DOCUMENT",
                      report["issue_counts"])

    def test_identical_source_checksum_conflicting_groups_block(self):
        checksum = "a" * 64
        corpus = [dict(self.corpus[0], source_group="source-family-a",
                       source_checksum_sha256=checksum),
                  dict(self.corpus[1], source_group="source-family-b",
                       source_checksum_sha256=checksum), *self.corpus[2:]]
        report = self.check(corpus=corpus)
        self.assertIn("CONFLICTING_SOURCE_GROUP_FOR_IDENTICAL_SOURCE",
                      report["issue_counts"])

    def test_bce_cross_tenant_negative_still_blocks(self):
        train = [dict(self.train[0], negative_chunk_ids=[4],
                      negative_text="another tenant delta")]
        report = self.check(train=train, loss_name="BinaryCrossEntropyLoss")
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("CROSS_TENANT_SOURCE", report["issue_counts"])


    def test_large_weak_supervision_audit_preserves_all_findings(self):
        train = [
            {"query_id": "t-" + str(i), "query_text": "compressor " + str(i),
             "tenant_id": 9, "chunk_id": 1, "positive_text": "compressor fault alpha"}
            for i in range(350)
        ]
        report = self.check(train=train, loss_name="BinaryCrossEntropyLoss")
        self.assertEqual(report["numbers"]["blockers"], 0)
        self.assertFalse(report["findings_truncated"])
        self.assertGreaterEqual(report["numbers"]["review_items"], 350)
        self.assertEqual(len(report["findings"]),
                         report["numbers"]["review_items"])

    def test_large_warning_queue_does_not_hide_cross_split_blocker(self):
        train = [
            {"query_id": "t-" + str(i), "query_text": "compressor " + str(i),
             "tenant_id": 9, "chunk_id": 1, "positive_text": "compressor fault alpha"}
            for i in range(350)
        ]
        dev = [dict(self.dev[0], negative_chunk_ids=[1],
                    negative_text="compressor fault alpha")]
        report = self.check(train=train, dev=dev, loss_name="BinaryCrossEntropyLoss")
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("SPLIT_DOCUMENT_COLLISION", report["issue_counts"])
        self.assertEqual(sum(f["severity"] == "BLOCKER" for f in report["findings"]),
                         report["numbers"]["blockers"])

if __name__ == "__main__":
    unittest.main()
