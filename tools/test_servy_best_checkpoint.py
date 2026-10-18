"""Offline contract tests for general embedding best-checkpoint selection."""
import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parent.parent / "core/ml/training.py"


def load_pure_best_checkpoint_guard():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "_best_validation_checkpoint")
    compiled = compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec")
    scope = {"math": math}
    exec(compiled, scope)
    return scope["_best_validation_checkpoint"]


class BestModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.guard = load_pure_best_checkpoint_guard()

    def test_finite_dev_checkpoint_is_explicit(self):
        state = SimpleNamespace(best_model_checkpoint="/research/checkpoint-4", best_metric=0.17)
        self.assertEqual(self.guard(SimpleNamespace(state=state), validation_required=True),
                         "/research/checkpoint-4")

    def test_absent_checkpoint_blocks(self):
        state = SimpleNamespace(best_model_checkpoint=None, best_metric=0.01)
        with self.assertRaisesRegex(RuntimeError, "no selected"):
            self.guard(SimpleNamespace(state=state), validation_required=True)

    def test_nan_metric_blocks(self):
        state = SimpleNamespace(best_model_checkpoint="/research/checkpoint-4", best_metric=float("nan"))
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            self.guard(SimpleNamespace(state=state), validation_required=True)

    def test_inf_metric_blocks(self):
        state = SimpleNamespace(best_model_checkpoint="/research/checkpoint-4", best_metric=float("inf"))
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            self.guard(SimpleNamespace(state=state), validation_required=True)

    def test_synthetic_smoke_allows_no_dev_checkpoint_without_claim(self):
        self.assertIsNone(self.guard(SimpleNamespace(state=None), validation_required=False))

    def test_training_export_uses_trainer_best_not_last_epoch(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        finetune = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "finetune")
        train_args = [n for n in ast.walk(finetune) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Name) and
                      n.func.id == "SentenceTransformerTrainingArguments"]
        self.assertEqual(len(train_args), 1)
        self.assertIn("load_best_model_at_end", [k.arg for k in train_args[0].keywords])
        exports = [n for n in ast.walk(finetune) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute) and n.func.attr == "save"]
        self.assertTrue(any(isinstance(n.func.value, ast.Attribute)
                            and n.func.value.attr == "model"
                            for n in exports))


if __name__ == "__main__":
    unittest.main()
