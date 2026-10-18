"""Synthetic-only tests of the no-approval Servy V4 artifact preparer."""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

from servy_v4_prepare_research_artifacts import (
    BGE_REVISION, build_bm25_baseline, freeze, make_config,
)
from servy_checkpoint_audit import load_jsonl, normalize_inputs, validate_candidates


class PrepareArtifactsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "private_training_data"
        self.root.mkdir()
        self.dev = self.root / "dev.jsonl"
        self.corpus = self.root / "corpus.jsonl"
        corpus = [
            {"chunk_id": "1", "tenant_id": "a", "document_id": "1",
             "passage": "maintain equipment blade cleaning lubrication"},
            {"chunk_id": "2", "tenant_id": "a", "document_id": "2",
             "passage": "network server restarts setup"},
            {"chunk_id": "3", "tenant_id": "b", "document_id": "3",
             "passage": "maintain equipment blade cleaning lubrication"},
        ]
        dev = [{"query_id": "q1", "tenant_id": "a",
                "query_text": "maintain blade lubrication",
                "positive_chunk_ids": ["2"]}]
        self.corpus.write_text("".join(json.dumps(x) + "\n" for x in corpus),
                               encoding="utf-8")
        self.dev.write_text("".join(json.dumps(x) + "\n" for x in dev),
                            encoding="utf-8")

    def test_real_commit_pin_and_loss(self):
        cfg = make_config()
        self.assertEqual(len(BGE_REVISION), 40)
        self.assertEqual(cfg["base_revision"], cfg["tokenizer_revision"])
        self.assertEqual(cfg["loss_name"], "BinaryCrossEntropyLoss")

    def test_bm25_does_not_force_gold_candidate_or_cross_tenant(self):
        cases, chunks = normalize_inputs(load_jsonl(self.dev), load_jsonl(self.corpus))
        ranked = build_bm25_baseline(cases, chunks, pool=10)
        self.assertEqual(ranked["q1"], ["1", "2"])
        self.assertNotIn("3", ranked["q1"])

    def test_writes_nonapproved_artifacts_and_rejects_overwrite(self):
        target = self.root / "frozen_artifacts"
        report = freeze(self.dev, self.corpus, target)
        self.assertEqual(report["status"], "NOT_APPROVED_FOR_TRAINING")
        self.assertEqual(report["evaluation_questions"], 1)
        self.assertFalse(report["human_review_complete"])
        self.assertEqual(report["candidate_method"], "bm25")
        self.assertTrue((target / "model.json").is_file())
        self.assertTrue((target / "frozen_candidates.jsonl").is_file())
        self.assertEqual(report["gold_candidate_coverage_diagnostic_only"]["present"], 1)
        with self.assertRaises(FileExistsError):
            freeze(self.dev, self.corpus, target)

    def test_qwen_export_freezes_actual_ranking_not_gold(self):
        source = self.root / "retrieval.jsonl"
        source.write_text(json.dumps({"query_id": "q1", "chunk_ids": ["2", "1"]}) + "\n")
        out = self.root / "qwen"
        report = freeze(self.dev, self.corpus, out,
                        method="qwen-export", retrieval_candidates=source)
        self.assertEqual(report["candidate_method"], "qwen-export")
        self.assertEqual(load_jsonl(out / "frozen_candidates.jsonl")[0]["chunk_ids"], ["2", "1"])

    def test_missing_proprietary_inputs_fail_closed(self):
        with self.assertRaises(FileNotFoundError):
            freeze(self.root / "missing.jsonl", self.corpus, self.root / "target")
        with self.assertRaisesRegex(ValueError, "private_training_data"):
            freeze(self.dev, self.corpus, Path(self.tmp.name) / "public")

    def test_rejects_cross_tenant_unsupported_qwen_export(self):
        source = self.root / "retrieval.jsonl"
        source.write_text(json.dumps({"query_id": "q1", "chunk_ids": ["3"]}) + "\n")
        with self.assertRaisesRegex(ValueError, "cross-tenant"):
            freeze(self.dev, self.corpus, self.root / "out",
                   method="qwen-export", retrieval_candidates=source)
        self.assertFalse((self.root / "out").exists())


if __name__ == "__main__":
    unittest.main()
