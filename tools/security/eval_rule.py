"""Find uses of Python's built-in eval() without flagging methods named eval.

PyTorch's ``model.eval()`` only switches a model to inference mode, so an
attribute call is flagged only when its owner is the builtins module.
"""
import ast

_BUILTINS = {"builtins", "__builtins__"}


def _is_builtins(node):
    return isinstance(node, ast.Name) and node.id in _BUILTINS


def builtin_eval_lines(tree):
    """Line numbers where the built-in eval is called or captured by reference."""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "eval":
            lines.append(node.lineno)              # eval(...), f = eval, map(eval, ...)
        elif isinstance(node, ast.Attribute) and node.attr == "eval" and _is_builtins(node.value):
            lines.append(node.lineno)              # builtins.eval(...)
        elif isinstance(node, ast.ImportFrom) and node.module == "builtins" and any(
                alias.name in ("eval", "*") for alias in node.names):
            lines.append(node.lineno)              # from builtins import eval as run
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
              and len(node.args) >= 2 and _is_builtins(node.args[0])
              and isinstance(node.args[1], ast.Constant) and node.args[1].value == "eval"):
            lines.append(node.lineno)              # getattr(builtins, "eval")
    return sorted(set(lines))
