"""Offline source-contract tests for Op26's provisional negative mining.

Reads source via AST only: never initializes Django, MySQL, or private data.
"""
import ast
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parent / "op26_build_training_dataset.py"


def isolated_screening_function():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "verify_negative_is_safe")
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"re": re}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["verify_negative_is_safe"]


class Op26LabelGovernanceTests(unittest.TestCase):
    def test_heading_difference_not_human_adjudication(self):
        screen = isolated_screening_function()
        positive = SimpleNamespace(
            text="Inspect pump relay before restart.", heading="Pump interlock")
        candidate = SimpleNamespace(
            text="Check relay and pump prior to power up.", heading="Control circuit")
        # These may be semantically equivalent; different headings and low token
        # overlap cannot establish that the candidate is irrelevant.
        self.assertTrue(screen("pump relay problem", positive, candidate))

    def test_output_negative_review_status_is_never_auto_approved(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        assignments = [
            node for node in ast.walk(tree) if isinstance(node, ast.Dict)
            and any(isinstance(key, ast.Constant) and
                    key.value == "negative_review_status" for key in node.keys)
        ]
        self.assertGreaterEqual(len(assignments), 2)
        for obj in assignments:
            for key, value in zip(obj.keys, obj.values):
                if isinstance(key, ast.Constant) and key.value == "negative_review_status":
                    self.assertEqual(ast.literal_eval(value), "provisional_unverified")

    def test_mining_function_never_sets_review_approval(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "find_hard_negatives")
        strings = [node.value for node in ast.walk(function)
                   if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        self.assertNotIn("approved", strings)


if __name__ == "__main__":
    unittest.main()
