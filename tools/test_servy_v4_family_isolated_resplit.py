"""Synthetic-only regression tests for fail-closed family-connected splits."""
import copy
import unittest

from servy_v4_family_isolated_resplit import Blocked, resplit


class TestV4FamilyResplit(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            {"chunk_id": str(i), "tenant_id": "6", "document_id": str(i),
             "source_version": "1", "source_checksum_sha256": str(i) * 64,
             "passage": "unique technical step " + str(i)}
            for i in range(1, 7)
        ]
        self.families = [
            {"tenant_id": "6", "document_id": str(i),
             "proposed_canonical_source_family": "A" if i <= 2 else
               "B" if i <= 4 else "C",
             "evidence_status": "REGISTRY_METADATA_MATCH"}
            for i in range(1, 7)
        ]
        self.train = [
            {"tenant_id": "6", "query_id": "q1", "query_text": "repair equipment A",
             "positive_chunk_ids": ["1"], "negative_chunk_ids": ["2"]},
            {"tenant_id": "6", "query_id": "q3", "query_text": "repair equipment C",
             "positive_chunk_ids": ["5"], "negative_chunk_ids": ["6"]},
        ]
        self.dev = [
            {"tenant_id": "6", "query_id": "q2", "query_text": "repair equipment B",
             "positive_chunk_ids": ["3"], "negative_chunk_ids": ["4"]}
        ]

    def test_isolates_families_without_dropping_queries(self):
        output, report = resplit(self.train, self.dev, self.corpus, self.families, [])
        self.assertEqual(report["status"], "PROVISIONAL_RESPLIT_ONLY")
        self.assertEqual(report["components"], 3)
        train, dev, quarantine = output
        self.assertEqual(len(train) + len(dev), 3)
        self.assertEqual({r["query_id"] for r in train + dev}, {"q1", "q2", "q3"})
        self.assertEqual(report["original_pair_counts"]["negative"], 3)
        self.assertFalse(report["training_ready"])

    def test_active_negatives_connect_all_families_and_block(self):
        train, dev = copy.deepcopy(self.train), copy.deepcopy(self.dev)
        train[0]["negative_chunk_ids"] = ["3"]
        dev[0]["negative_chunk_ids"] = ["5"]
        result, report = resplit(train, dev, self.corpus, self.families, [])
        self.assertIsNone(result)
        self.assertEqual(report["status"], "BLOCKED_SINGLE_CONNECTED_COMPONENT")
        self.assertEqual(report["component_sizes_descending"], [3])

    def test_quarantine_ledger_preserved_and_moved(self):
        train = copy.deepcopy(self.train)
        train[0]["quarantined_negative_chunk_ids"] = ["4"]
        ledger = [{"split": "train", "query_id": "q1",
                   "negative_chunk_id": "4", "review_status": "PENDING"}]
        output, report = resplit(train, self.dev, self.corpus, self.families, ledger)
        t, d, q = output
        owner = "dev" if any(row["query_id"] == "q1" for row in d) else "train"
        self.assertEqual(q[0]["split"], owner)
        self.assertEqual(q[0]["review_status"], "PENDING")
        self.assertEqual(report["quarantined_negatives_preserved"], 1)

    def test_missing_family_or_cross_tenant_source_blocks(self):
        with self.assertRaisesRegex(Blocked, "UNCOVERED_DOCUMENT_FAMILY"):
            resplit(self.train, self.dev, self.corpus, self.families[:-1], [])
        train = copy.deepcopy(self.train)
        train[0]["negative_chunk_ids"] = ["999"]
        with self.assertRaisesRegex(Blocked, "MISSING_OR_CROSS_TENANT_CHUNK"):
            resplit(train, self.dev, self.corpus, self.families, [])

    def test_cross_tenant_identical_passage_is_connected(self):
        corpus, families, train = (copy.deepcopy(self.corpus),
                                   copy.deepcopy(self.families),
                                   copy.deepcopy(self.train))
        corpus[-1]["tenant_id"] = "9"
        corpus[-1]["passage"] = corpus[0]["passage"]
        families[-1]["tenant_id"] = "9"
        train[-1]["tenant_id"] = "9"
        train[-1]["positive_chunk_ids"] = ["6"]
        train[-1]["negative_chunk_ids"] = []
        output, report = resplit(train, self.dev, corpus, families, [])
        self.assertEqual(report["components"], 2)
        self.assertEqual(report["total_queries"], 3)

if __name__ == "__main__":
    unittest.main()
