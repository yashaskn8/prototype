"""No-model, no-Django contract tests for the old Op27 experimental trainer."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

from op27_passages import op27_passage_text


SCRIPT = Path(__file__).with_name("op27_train_and_evaluate.py")


def function_source(name):
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == name]
    if len(found) != 1:
        raise AssertionError(f"Expected exactly one {name} function")
    return ast.get_source_segment(source, found[0])


class Op27TrainingReadinessTests(unittest.TestCase):
    def setUp(self):
        document = SimpleNamespace(
            title="Compressor Repair Guide", description="Maintenance reference",
            tags="pump", doc_type="repair_manual",
            domain=None, category=None, product=None,
            asset=SimpleNamespace(name="Compressor", model_number="C-4"),
        )
        self.chunk = SimpleNamespace(
            heading="Pressure fault",
            text="Reduce pressure and inspect valve.",
            document=document,
        )

    def test_legacy_index_text_contains_document_metadata(self):
        result = op27_passage_text(self.chunk, content_first=False)
        self.assertIn("Compressor Repair Guide", result)
        self.assertIn("C-4", result)
        self.assertIn("Pressure fault", result)
        self.assertIn("Reduce pressure", result)

    def test_content_first_has_no_doc_title(self):
        result = op27_passage_text(self.chunk, content_first=True)
        self.assertEqual(result, "Pressure fault\nReduce pressure and inspect valve.")

    def test_content_first_does_not_duplicate_embedded_heading(self):
        self.chunk.text = "Pressure fault\nInspect valve."
        self.assertEqual(op27_passage_text(self.chunk, content_first=True),
                         self.chunk.text)

    def test_train_and_index_share_one_formatter(self):
        self.assertIn("op27_passage_text(", function_source("resolve_chunk_texts"))
        self.assertIn("op27_passage_text(", function_source("build_and_stamp_chroma_collection"))

    def test_best_validation_model_selected_instead_of_last_epoch(self):
        src = function_source("train_single_model")
        self.assertIn("load_best_model_at_end=True", src)
        self.assertIn('metric_for_best_model="eval_loss"', src)
        self.assertIn("greater_is_better=False", src)
        self.assertIn('"positive": [r["positive_text"] for r in val_resolved]', src)
        self.assertIn("len(val_resolved) < 8", src)

    def test_training_and_index_overwrite_protected(self):
        self.assertIn('if (output_dir / "final").exists():',
                      function_source("train_single_model"))
        index_src = function_source("build_and_stamp_chroma_collection")
        self.assertIn("raise FileExistsError", index_src)
        self.assertNotIn("client.delete_collection(collection_name)", index_src)

    def test_missing_source_chunks_are_training_errors(self):
        src = function_source("resolve_chunk_texts")
        self.assertIn("missing training chunk IDs", src)
        self.assertIn("cross-tenant chunk", src)
        self.assertIn("passage labeled both positive and negative", src)


if __name__ == "__main__":
    unittest.main()
