import ast
import unittest

from tools.security.eval_rule import builtin_eval_lines


def lines(source):
    return builtin_eval_lines(ast.parse(source))


class BuiltinEvalRuleTests(unittest.TestCase):
    def test_flags_the_builtin_in_every_spelling(self):
        for source in (
            "eval(user_text)",
            "result = eval('1 + 1')",
            "import builtins\nbuiltins.eval(x)",
            "__builtins__.eval(x)",
            "from builtins import eval as run\nrun(x)",
            "from builtins import *",
            "import builtins\ngetattr(builtins, 'eval')(x)",
            "runner = eval\nrunner(x)",
            "list(map(eval, rows))",
        ):
            with self.subTest(source=source):
                self.assertTrue(lines(source))

    def test_model_eval_methods_are_not_flagged(self):
        for source in (
            "model.eval()",
            "model = AutoModel.from_pretrained(path).eval()",
            "self.encoder.eval()",
            "getattr(model, 'eval')()",
            "name = 'eval'",
            "def evaluate(x):\n    return x",
        ):
            with self.subTest(source=source):
                self.assertEqual(lines(source), [])

    def test_reports_line_numbers(self):
        self.assertEqual(lines("x = 1\nmodel.eval()\neval(x)\n"), [3])

    def test_current_onnx_export_is_not_flagged(self):
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / "core" / "management" / "commands" / "export_v5_onnx.py"
        self.assertEqual(builtin_eval_lines(ast.parse(path.read_text(encoding="utf-8"))), [])


if __name__ == "__main__":
    unittest.main()
