"""Offline synthetic tests; never read company documents."""
import json
import tempfile
import unittest
from pathlib import Path
from servy_v4_private_reconciliation import reconcile, process


class TestReconcile(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            {"chunk_id": "1", "tenant_id": "9", "document_id": "10",
             "passage": "Safe compressor fix", "source_version": "1",
             "source_checksum_sha256": "a"*64},
            {"chunk_id": "2", "tenant_id": "9", "document_id": "20",
             "passage": "Calibrate sensor", "source_version": "1",
             "source_checksum_sha256": "b"*64},
            {"chunk_id": "3", "tenant_id": "9", "document_id": "30",
             "passage": "Filter cleaning", "source_version": "1",
             "source_checksum_sha256": "c"*64},
        ]
        self.train = [{"tenant_id": "9", "query_id": "t1", "query_text": "compressor",
                       "positive_chunk_ids": ["1"], "negative_chunk_ids": ["3"],
                       "source_group": "manual-one",
                       "review_status": "weakly_supervised"}]
        self.dev = [{"tenant_id": "9", "query_id": "d1", "query_text": "sensor",
                     "positive_chunk_ids": ["2"], "negative_chunk_ids": [],
                     "source_group": "manual-two",
                     "review_status": "weakly_supervised"}]

    def test_clean_structural_candidate_stays_unapproved(self):
        report, proposed, review, families = reconcile(
            self.train, self.dev, self.corpus)
        self.assertTrue(report["quarantine_ledger_matches_split_proposals"])
        self.assertEqual(report["unapproved_negative_rows"], 1)
        self.assertEqual(report["train_dev_overlap"]["documents"], 0)
        self.assertEqual(report["status"], "BLOCKED_SOURCE_FAMILY_VERIFICATION")
        self.assertEqual(report["unapproved_label_rows"], 2)
        self.assertFalse(report["ready_for_training"])
        self.assertEqual(proposed["train"][0]["positive_texts"], ["Safe compressor fix"])
        self.assertEqual(proposed["train"][0]["negative_texts"], ["Filter cleaning"])
        self.assertEqual(proposed["train"][0]["review_status"], "weakly_supervised")
        self.assertEqual(len(review), 2)
        self.assertTrue(all(ref["decision"] == "PENDING"
                            for row in review for ref in row["references"]))
        self.assertTrue(all(f["decision"] == "PENDING" and
                            f["verified_canonical_source_family"] is None
                            for f in families))

    def test_negative_bridge_blocks(self):
        rows = [dict(self.train[0], negative_chunk_ids=["2"])]
        report, _, _, _ = reconcile(rows, self.dev, self.corpus)
        self.assertEqual(report["train_dev_overlap"]["documents"], 1)
        self.assertEqual(report["status"], "BLOCKED_DATA_INTEGRITY")

    def test_same_passage_different_document_leaks(self):
        corpus = list(self.corpus)
        corpus[1] = dict(corpus[1], passage=" safe COMPRESSOR fix ")
        report, _, _, _ = reconcile(self.train, self.dev, corpus)
        self.assertEqual(report["train_dev_overlap"]["identical_passages"], 1)

    def test_same_document_group_conflict_flagged_not_fabricated(self):
        train = self.train + [dict(self.train[0], query_id="t2",
                                   query_text="different question",
                                   source_group="different-family")]
        report, _, _, families = reconcile(train, self.dev, self.corpus)
        self.assertEqual(report["conflicting_declared_document_groups"], 1)
        self.assertTrue(any(f["conflicting_group_claims"] for f in families))

    def test_missing_positive_and_cross_tenant_block(self):
        train = [dict(self.train[0], positive_chunk_ids=["999"])]
        report, _, _, _ = reconcile(train, self.dev, self.corpus)
        self.assertIn("MISSING_CHUNK_REFERENCE", report["issue_counts"])
        wrong = [dict(self.train[0], tenant_id="8")]
        report, _, _, _ = reconcile(wrong, self.dev, self.corpus)
        self.assertIn("CROSS_TENANT_REFERENCE", report["issue_counts"])

    def test_historical_holdout_missing_gold_still_not_independent(self):
        holdout = [{"tenant_id": "9", "positive_chunk_ids": ["999", "1"]}]
        report, _, _, _ = reconcile(self.train, self.dev, self.corpus,
                                   historical_holdout=holdout)
        self.assertEqual(report["historical_holdout"]["unresolved_positive_chunk_ids"], 1)
        self.assertEqual(report["historical_holdout"]["shared_train_documents"], 1)
        self.assertFalse(report["historical_holdout"]["independent"])

    def test_local_private_output_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "private_training_data"
            proposal = p / "recovery_v4" / "split_proposal_1"
            proposal.mkdir(parents=True)
            corpus_file = p / "corpus.jsonl"
            for path, rows in (
                (proposal / "train_proposed.jsonl", [
                    dict(self.train[0], quarantined_negative_chunk_ids=["2"])]),
                (proposal / "dev_proposed.jsonl", self.dev),
                (proposal / "quarantined_negatives.jsonl", [
                    {"split": "train", "query_id": "t1",
                     "negative_chunk_id": "2", "review_status": "PENDING"}
                ]),
                (corpus_file, self.corpus),
            ):
                path.write_text("".join(json.dumps(x)+"\n" for x in rows),
                                encoding="utf-8")
            target = p / "recovery_v4" / "review_bundle"
            report = process(proposal, corpus_file, None, target)
            self.assertFalse(report["ready_for_training"])
            self.assertTrue((target / "readiness_report.json").exists())
            with self.assertRaises(FileExistsError):
                process(proposal, corpus_file, None, target)
            with self.assertRaisesRegex(ValueError, "private_training_data"):
                process(proposal, corpus_file, None, Path(tmp) / "public")

    def test_quarantine_ledger_matches_proposal(self):
        train = [dict(self.train[0], quarantined_negative_chunk_ids=["2"])]
        quarantine = [{"split": "train", "query_id": "t1",
                       "negative_chunk_id": "2", "review_status": "PENDING"}]
        report, proposed, _, _ = reconcile(
            train, self.dev, self.corpus, quarantine=quarantine)
        self.assertTrue(report["quarantine_ledger_matches_split_proposals"])
        self.assertNotIn("QUARANTINE_LEDGER_MISSING_REFERENCES", report["issue_counts"])
        self.assertEqual(proposed["train"][0]["quarantined_negative_chunk_ids"], ["2"])

    def test_missing_ledger_reference_blocks(self):
        train = [dict(self.train[0], quarantined_negative_chunk_ids=["2"])]
        report, _, _, _ = reconcile(train, self.dev, self.corpus, quarantine=[])
        self.assertEqual(report["status"], "BLOCKED_DATA_INTEGRITY")
        self.assertIn("QUARANTINE_LEDGER_MISSING_REFERENCES", report["issue_counts"])

    def test_fake_or_duplicate_ledger_reference_blocks(self):
        train = [dict(self.train[0], quarantined_negative_chunk_ids=["2"])]
        quarantine = [{"split": "train", "query_id": "t1",
                       "negative_chunk_id": "2", "review_status": "PENDING"}]
        for ledger in (quarantine*2,
                       quarantine + [{"split": "dev", "query_id": "d1",
                                      "negative_chunk_id": "3"}],
                       [{"split": "train", "query_id": "t1",
                         "negative_chunk_id": "999"}]):
            report, _, _, _ = reconcile(train, self.dev, self.corpus, quarantine=ledger)
            self.assertEqual(report["status"], "BLOCKED_DATA_INTEGRITY")
            self.assertFalse(report["quarantine_ledger_matches_split_proposals"])

    def test_quarantined_reference_cannot_remain_active(self):
        train = [dict(self.train[0], quarantined_negative_chunk_ids=["3"])]
        quarantine = [{"split": "train", "query_id": "t1",
                       "negative_chunk_id": "3"}]
        report, _, _, _ = reconcile(train, self.dev, self.corpus, quarantine=quarantine)
        self.assertIn("QUARANTINED_NEGATIVE_STILL_ACTIVE", report["issue_counts"])
        self.assertEqual(report["status"], "BLOCKED_DATA_INTEGRITY")

    def test_no_autopromotion_of_negative_approval(self):
        train = [dict(self.train[0], review_status="approved",
                      reviewer_id="reviewer-1")]
        report, proposed, _, _ = reconcile(train, self.dev, self.corpus)
        self.assertEqual(report["unapproved_negative_rows"], 1)
        self.assertFalse(report["expert_label_verification_complete"])
        self.assertFalse(report["ready_for_training"])
        self.assertNotIn("negative_review_status", proposed["train"][0])


    def test_cross_tenant_identical_checksum_detected(self):
        train = [dict(self.train[0], negative_chunk_ids=[])]
        dev = [dict(self.dev[0], tenant_id="8", query_text="different question")]
        corpus = [dict(c) for c in self.corpus]
        corpus[1]["tenant_id"] = "8"
        corpus[1]["source_checksum_sha256"] = corpus[0]["source_checksum_sha256"]
        report, _, _, _ = reconcile(train, dev, corpus)
        self.assertEqual(report["train_dev_overlap"]["documents"], 0)
        self.assertEqual(report["train_dev_overlap"]["identical_checksums_across_tenants"], 1)
        self.assertEqual(report["status"], "BLOCKED_DATA_INTEGRITY")

    def test_cross_tenant_identical_passage_detected(self):
        train = [dict(self.train[0], negative_chunk_ids=[])]
        dev = [dict(self.dev[0], tenant_id="8", query_text="different question")]
        corpus = [dict(c) for c in self.corpus]
        corpus[1]["tenant_id"] = "8"
        corpus[1]["passage"] = corpus[0]["passage"]
        report, _, _, _ = reconcile(train, dev, corpus)
        self.assertEqual(report["train_dev_overlap"]["identical_passages_across_tenants"], 1)

