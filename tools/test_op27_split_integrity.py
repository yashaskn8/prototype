"""No-model contract tests for strict Op27 train-dev group isolation."""
import unittest

from op27_split_integrity import enforce_train_dev_separation


class Op27SplitTests(unittest.TestCase):
    def setUp(self):
        self.train = [
            dict(query_id="tr-1", query_text="Fault alpha?", tenant_id=9,
                 source_group="guide-alpha", positive_document_ids=[10],
                 negative_document_ids=[11], positive_chunk_ids=[100],
                 negative_chunk_ids=[110], positive_text="Check alpha valve",
                 negative_text="Check beta motor"),
        ]
        self.dev = [
            dict(query_id="dv-1", query_text="Fault gamma?", tenant_id=9,
                 source_group="guide-gamma", positive_document_ids=[30],
                 negative_document_ids=[31], positive_chunk_ids=[300],
                 negative_chunk_ids=[310], positive_text="Check gamma sensor",
                 negative_text="Check delta filter"),
        ]

    def test_genuinely_disjoint(self):
        result = enforce_train_dev_separation(self.train, self.dev)
        self.assertEqual(result["structural_overlap_count"], 0)
        self.assertFalse(result["human_truth_verified"])

    def test_normalized_duplicate_query_is_blocked(self):
        self.dev[0]["query_text"] = " Fault    ALPHA!! "
        with self.assertRaisesRegex(ValueError, "same_query"):
            enforce_train_dev_separation(self.train, self.dev)

    def test_training_negative_source_becomes_dev_positive(self):
        self.dev[0]["positive_document_ids"] = [11]
        with self.assertRaisesRegex(ValueError, "same_source_document"):
            enforce_train_dev_separation(self.train, self.dev)

    def test_training_negative_chunk_becomes_dev_positive(self):
        self.dev[0]["positive_chunk_ids"] = [110]
        with self.assertRaisesRegex(ValueError, "negative_matches_dev_positive"):
            enforce_train_dev_separation(self.train, self.dev)

    def test_training_positive_becomes_dev_negative(self):
        self.dev[0]["negative_chunk_ids"] = [100]
        with self.assertRaisesRegex(ValueError, "dev_negative_matches_training_positive"):
            enforce_train_dev_separation(self.train, self.dev)

    def test_changed_ids_same_passage_detected(self):
        self.dev[0]["positive_text"] = "  CHECK  ALPHA VALVE "
        with self.assertRaisesRegex(ValueError, "same_positive_passage"):
            enforce_train_dev_separation(self.train, self.dev)

    def test_changed_ids_same_source_group_detected(self):
        self.dev[0]["source_group"] = " GUIDE Alpha "
        with self.assertRaisesRegex(ValueError, "same_source_group"):
            enforce_train_dev_separation(self.train, self.dev)

    def test_unknown_source_docs_block(self):
        del self.dev[0]["positive_document_ids"]
        del self.dev[0]["negative_document_ids"]
        with self.assertRaisesRegex(ValueError, "source document identity missing"):
            enforce_train_dev_separation(self.train, self.dev)

    def test_missing_query_or_tenant_blocked(self):
        self.train[0]["tenant_id"] = ""
        with self.assertRaisesRegex(ValueError, "missing query_id, tenant or query"):
            enforce_train_dev_separation(self.train, self.dev)

    def test_missing_eval_blocked(self):
        with self.assertRaisesRegex(ValueError, "dataset is empty"):
            enforce_train_dev_separation(self.train, [])

    def test_duplicate_qid_inside_train_blocked(self):
        self.train.append(dict(self.train[0]))
        with self.assertRaisesRegex(ValueError, "duplicate query ID"):
            enforce_train_dev_separation(self.train, self.dev)


if __name__ == "__main__":
    unittest.main()
