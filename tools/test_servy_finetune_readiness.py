"""Offline synthetic tests of fine-tuning readiness. No models, DB, Docker."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from servy_finetune_readiness import (
    assess, build_corpus, inspect_split, load_rows,
)


class FineTuneReadinessTests(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            dict(chunk_id=1, tenant_id=9, document_id=11, passage="Equipment Alpha pressure valve repair"),
            dict(chunk_id=2, tenant_id=9, document_id=22, passage="Equipment Beta motor is running hot"),
            dict(chunk_id=3, tenant_id=9, document_id=33, passage="Equipment Gamma temperature sensor wiring"),
            dict(chunk_id=4, tenant_id=9, document_id=44, passage="Equipment Delta diagnosis procedure"),
        ]
        self.train = [
            dict(query_id="tr1", query_text="Repair Alpha pressure?", tenant_id=9,
                 chunk_id=1, positive_text="Equipment Alpha pressure valve repair",
                 negative_chunk_ids=[2], source_group="doc-11",
                 review_status="weakly_supervised"),
        ]
        self.dev = [
            dict(query_id="dv1", query_text="Check Gamma sensor?", tenant_id=9,
                 chunk_id=3, positive_text="Equipment Gamma temperature sensor wiring",
                 source_group="doc-33", review_status="weakly_supervised"),
        ]
        self.stage2 = [
            dict(query_id="s2a", query="Repair Alpha pressure?",
                 positive="Equipment Alpha pressure valve repair",
                 candidate_negative="Equipment Beta motor is running hot",
                 expert_signoff="", approved_for_training=""),
        ]

    def test_structurally_eligible_research_is_not_expert_verified(self):
        report = assess(self.train, self.dev, self.corpus, stage2=self.stage2)
        self.assertEqual(report["status"], "RESEARCH_PREFLIGHT_PASSED")
        self.assertFalse(report["production_eligible"])
        self.assertEqual(report["stage2"]["expert_approved"], 0)
        self.assertTrue(any("expert approval" in s for s in report["review_findings"]))

    def test_stage2_negative_source_in_dev_is_blocked(self):
        self.stage2[0]["candidate_negative"] = self.corpus[2]["passage"]
        report = assess(self.train, self.dev, self.corpus, stage2=self.stage2)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any("source document" in s for s in report["blocking_findings"]))

    def test_different_query_same_guide_leaks(self):
        self.dev[0]["chunk_id"] = 1
        self.dev[0]["positive_text"] = self.corpus[0]["passage"]
        self.dev[0]["source_group"] = "doc-11"
        report = assess(self.train, self.dev, self.corpus)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any("shared" in s or "collision" in s
                            for s in report["blocking_findings"]))

    def test_unnamed_stage2_passage_is_rejected(self):
        self.stage2[0]["candidate_negative"] = "unsupported nonindexed passage"
        report = assess(self.train, self.dev, self.corpus, stage2=self.stage2)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any("untraceable/ambiguous negative" in s
                            for s in report["blocking_findings"]))

    def test_positive_label_cannot_be_negative(self):
        self.stage2[0]["candidate_negative"] = self.corpus[0]["passage"]
        report = assess(self.train, self.dev, self.corpus, stage2=self.stage2)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any("positive passage incorrectly labeled" in s
                            for s in report["blocking_findings"]))

    def test_same_question_cross_splits_is_blocked(self):
        self.dev[0]["query_text"] = "  REPAIR Alpha pressure!! "
        report = assess(self.train, self.dev, self.corpus)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any("duplicate normalized question" in s
                            for s in report["blocking_findings"]))

    def test_cross_tenant_chunk_is_rejected(self):
        self.corpus[2]["tenant_id"] = 8
        report = assess(self.train, self.dev, self.corpus)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any("cross-tenant" in s for s in report["blocking_findings"]))

    def test_unknown_training_positive_blocks(self):
        self.train[0]["chunk_id"] = 999
        report = assess(self.train, self.dev, self.corpus)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any("unknown positive chunk" in s for s in report["blocking_findings"]))

    def test_same_document_negative_warns_not_auto_verifies(self):
        self.corpus[1]["document_id"] = 11
        report = assess(self.train, self.dev, self.corpus)
        self.assertEqual(report["status"], "RESEARCH_PREFLIGHT_PASSED")
        self.assertTrue(any("same-document hard negative" in s
                            for s in report["review_findings"]))

    def test_missing_gold_in_holdout_blocks_certification(self):
        holdout = [dict(query_id="h1", query="Holdout technician question",
                        tenant_id=9, positive_chunk_ids=[700])]
        report = assess(self.train, self.dev, self.corpus, holdout=holdout)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any("gold ID not in exported corpus" in s
                            for s in report["blocking_findings"]))

    def test_unanswerable_holdout_only_checks_query_reuse(self):
        holdout = [dict(query_id="h1", query="Repair Alpha pressure?", tenant_id=9,
                        positive_chunk_ids=[])]
        report = assess(self.train, self.dev, self.corpus, holdout=holdout)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any("duplicate normalized question" in s
                            for s in report["blocking_findings"]))

    def test_unknown_or_duplicate_corpus_chunk_blocked(self):
        with self.assertRaisesRegex(ValueError, "Duplicated corpus"):
            build_corpus(self.corpus + [self.corpus[0]])

    def test_read_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "data.jsonl"
            file.write_text('{"query_id":"a"}\n', encoding="utf-8")
            self.assertEqual(load_rows(file)[0]["query_id"], "a")

    def test_does_not_write_training_or_labels(self):
        before = json.dumps([self.train, self.dev, self.corpus, self.stage2], sort_keys=True)
        assess(self.train, self.dev, self.corpus, stage2=self.stage2)
        after = json.dumps([self.train, self.dev, self.corpus, self.stage2], sort_keys=True)
        self.assertEqual(before, after)

    def test_dev_negative_from_train_source_blocks(self):
        self.dev[0]["negative_chunk_ids"] = [2]
        report = assess(self.train, self.dev, self.corpus)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any("negative source document also appears" in x
                            for x in report["blocking_findings"]))

    def test_stage2_must_use_known_train_questions(self):
        self.stage2[0]["query"] = "New unsourced technician question"
        report = assess(self.train, self.dev, self.corpus, stage2=self.stage2)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any("question not found in Stage1 training" in x
                            for x in report["blocking_findings"]))


if __name__ == "__main__":
    unittest.main()
