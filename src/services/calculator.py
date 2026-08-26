"""
A restricted, `eval`-free arithmetic expression evaluator.

Split out into its own module (mirroring `hybrid_retrieval.py`'s split from `retrieval_service.py`)
so it stays unit-testable without importing `agent_tools.py`, which pulls in `retrieval_service`
and its real embedding/cross-encoder model loads at import time.
"""
import ast
import operator
from typing import Callable, Dict, Type

_ALLOWED_BINOPS: Dict[Type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS: Dict[Type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Expressions are ultimately LLM-influenced input (directly from the model's tool-call args, and
# indirectly from retrieved filing text a prompt-injected chunk could try to steer it with), so
# both operand and result magnitudes are bounded - not for numeric-range reasons, but so a
# pathological expression (e.g. a huge `**` exponent, or a long chain of multiplications) can't
# make Python spend a long time / a lot of memory materializing an enormous int synchronously
# inside a request.
_MAX_MAGNITUDE = 1e15
_MAX_EXPONENT_MAGNITUDE = 1000


def _check_magnitude(value: float) -> float:
    if abs(value) > _MAX_MAGNITUDE:
        raise ValueError(f"Result magnitude too large (limit {_MAX_MAGNITUDE:g}): {value}")
    return value


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a restricted arithmetic AST node. Only numeric literals, the binary
    operators in `_ALLOWED_BINOPS`, and the unary operators in `_ALLOWED_UNARYOPS` are permitted -
    any other node type (names, calls, attributes, comprehensions, etc.) raises `ValueError`.
    Every literal and intermediate result is also magnitude-checked (see `_MAX_MAGNITUDE`), and
    `**` additionally has its exponent bounded before it's evaluated (see
    `_MAX_EXPONENT_MAGNITUDE`), so no single node can produce or require computing a pathologically
    large number. Deliberately never uses `eval`/`exec`, since the expression is ultimately
    LLM-influenced input.
    """
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return _check_magnitude(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT_MAGNITUDE:
            raise ValueError(
                f"Exponent magnitude too large (limit {_MAX_EXPONENT_MAGNITUDE}): {right}"
            )
        return _check_magnitude(_ALLOWED_BINOPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _check_magnitude(_ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand)))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def safe_eval(expression: str) -> float:
    """
    Evaluate a numeric arithmetic expression without ever calling `eval`/`exec`.

    Args:
        expression: An arithmetic expression using only numeric literals, `+ - * / % **`, and
            parentheses.

    Returns:
        The numeric result.

    Raises:
        ValueError: If `expression` doesn't parse as a restricted arithmetic expression, contains
            any disallowed construct (names, calls, attribute access, etc.), any literal,
            exponent, or intermediate/final result exceeds the magnitude bounds in
            `_MAX_MAGNITUDE`/`_MAX_EXPONENT_MAGNITUDE`, or is nested too deeply (e.g. a long chain
            of unary operators or parentheses) to parse/evaluate within Python's recursion limit.
        SyntaxError: If `expression` isn't valid Python expression syntax at all.
        ZeroDivisionError: If the expression divides by zero.
    """
    try:
        parsed = ast.parse(expression, mode="eval")
        return _eval_node(parsed.body)
    except RecursionError:
        raise ValueError("Expression is nested too deeply to evaluate") from None
