"""Pure synthetic tests; no MySQL, Docker, torch or private files."""
import unittest
from types import SimpleNamespace
from op26_source_isolated_split import source_isolated_split


def c(tenant, doc, passage, sha=""):
    return SimpleNamespace(
        tenant_id=tenant, document_id=doc, text=passage,
        document=SimpleNamespace(checksum_sha256=sha))


class SourceGraphSplitTests(unittest.TestCase):
    def setUp(self):
        self.chunks = {
            1: c(9, 10, "Compressor"),
            2: c(9, 20, "Temperature"),
            3: c(9, 30, "Filter"),
            4: c(9, 40, "Valve"),
        }

    def row(self, name, pos, neg=None):
        return {"tenant_id": 9, "query_id": name, "query_text": name,
                "positive_chunk_ids": [pos], "negative_chunk_ids": neg or [],
                "source_group": "misleading-positive-only-group-" + name}

    def test_negative_bridge_never_crosses_split(self):
        train, dev, evidence = source_isolated_split([
            self.row("A", 1, [2]), self.row("B", 2, [3]),
            self.row("C", 4)], self.chunks)
        lookup = {r["query_id"]: "train" for r in train}
        lookup.update({r["query_id"]: "dev" for r in dev})
        self.assertEqual(lookup["A"], lookup["B"])
        self.assertNotEqual(lookup["A"], lookup["C"])
        self.assertEqual(evidence["source_footprint_overlap"], 0)

    def test_identical_document_checksum_not_split(self):
        self.chunks[5] = c(9, 50, "Text A", "a"*64)
        self.chunks[6] = c(9, 60, "Text B", "a"*64)
        train, dev, _ = source_isolated_split([
            self.row("A", 5), self.row("B", 6), self.row("C", 4)], self.chunks)
        lookup = {r["query_id"]: "train" for r in train}
        lookup.update({r["query_id"]: "dev" for r in dev})
        self.assertEqual(lookup["A"], lookup["B"])

    def test_copied_passage_not_split(self):
        self.chunks[5] = c(9, 50, "   COMPRESSOR  ")
        train, dev, _ = source_isolated_split([
            self.row("A", 1), self.row("B", 5), self.row("C", 4)], self.chunks)
        lookup = {r["query_id"]: "train" for r in train}
        lookup.update({r["query_id"]: "dev" for r in dev})
        self.assertEqual(lookup["A"], lookup["B"])

    def test_all_connected_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "one connected component"):
            source_isolated_split([
                self.row("A", 1, [2]), self.row("B", 2)], self.chunks)

    def test_unknown_source_and_tenant_mismatch_fail(self):
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            source_isolated_split([self.row("A", 999), self.row("B", 2)], self.chunks)
        bad = self.row("A", 1)
        bad["tenant_id"] = 8
        with self.assertRaisesRegex(ValueError, "Cross-tenant"):
            source_isolated_split([bad, self.row("B", 2)], self.chunks)

    def test_repeatable(self):
        rows = [self.row("A", 1), self.row("B", 2), self.row("C", 3)]
        self.assertEqual(source_isolated_split(rows, self.chunks),
                         source_isolated_split(rows, self.chunks))


    def test_identical_manual_copies_across_tenants_share_component(self):
        self.chunks[5] = c(8, 50, "  COMPRESSOR ")
        rows = [self.row("A", 1), self.row("C", 4),
                {"tenant_id": 8, "query_id": "B", "query_text": "Different question",
                 "positive_chunk_ids": [5]}]
        train, dev, _ = source_isolated_split(rows, self.chunks)
        members = {r["query_id"]: "train" for r in train}
        members.update({r["query_id"]: "dev" for r in dev})
        self.assertEqual(members["A"], members["B"])

    def test_identical_source_sha_across_tenants_share_component(self):
        self.chunks[5] = c(8, 50, "Different passage", "d"*64)
        self.chunks[6] = c(9, 60, "Another passage", "d"*64)
        rows = [self.row("A", 6), self.row("C", 4),
                {"tenant_id": 8, "query_id": "B", "query_text": "Different question",
                 "positive_chunk_ids": [5]}]
        train, dev, _ = source_isolated_split(rows, self.chunks)
        members = {r["query_id"]: "train" for r in train}
        members.update({r["query_id"]: "dev" for r in dev})
        self.assertEqual(members["A"], members["B"])

if __name__ == "__main__":
    unittest.main()
