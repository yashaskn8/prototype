"""Public synthetic model-free validation; no Hub downloads or CUDA in CI."""
import ast
import unittest
from pathlib import Path
from servy_v4_colab_smoke import synthetic_pairs, validate_pairs, PUBLIC_TINY_MODEL


class SmokeContracts(unittest.TestCase):
    def test_training_and_development_both_have_binary_labels(self):
        train, dev = synthetic_pairs()
        validate_pairs(train, dev)
        self.assertEqual({r["label"] for r in train}, {0.0, 1.0})
        self.assertEqual({r["label"] for r in dev}, {0.0, 1.0})

    def test_rejects_malformed_training_schema(self):
        train, dev = synthetic_pairs()
        train[0]["unexpected"] = "extra"
        with self.assertRaisesRegex(ValueError, "dataset columns"):
            validate_pairs(train, dev)

    def test_smoke_has_no_servy_private_file_inputs(self):
        source = Path(__file__).with_name("servy_v4_colab_smoke.py").read_text(
            encoding="utf-8")
        tree = ast.parse(source)
        # All load-bearing inputs are fixed public synthetic pairs.
        self.assertNotIn("private_training_data", source)
        self.assertNotIn("db.sqlite3", source)
        self.assertNotIn("drive.mount", source)
        self.assertIn("hf-internal-testing/tiny-random-bert", source)
        self.assertEqual(PUBLIC_TINY_MODEL, "hf-internal-testing/tiny-random-bert")

    def test_rejects_one_class_validation(self):
        train, dev = synthetic_pairs()
        with self.assertRaisesRegex(ValueError, "both relevance classes"):
            validate_pairs(train, dev[:1])


if __name__ == "__main__":
    unittest.main()
