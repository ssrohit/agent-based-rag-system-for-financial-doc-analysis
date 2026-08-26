import pytest

from src.services.calculator import safe_eval


def test_safe_eval_basic_arithmetic():
    assert safe_eval("2 + 3 * 4") == 14


def test_safe_eval_parentheses_and_precedence():
    assert safe_eval("(2 + 3) * 4") == 20


def test_safe_eval_percent_change_expression():
    result = safe_eval("(123.4 - 100.2) / 100.2 * 100")
    assert result == pytest.approx(23.1536926, rel=1e-6)


def test_safe_eval_unary_minus():
    assert safe_eval("-5 + 2") == -3


def test_safe_eval_power_and_modulo():
    assert safe_eval("2 ** 10") == 1024
    assert safe_eval("10 % 3") == 1


def test_safe_eval_division_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        safe_eval("1 / 0")


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",
        "open('x').read()",
        "1 and 2",
        "[1, 2, 3]",
        "x + 1",
        "len('abc')",
    ],
)
def test_safe_eval_rejects_non_arithmetic_constructs(expression):
    with pytest.raises((ValueError, SyntaxError)):
        safe_eval(expression)


def test_safe_eval_rejects_oversized_exponent_without_computing_it():
    # A huge exponent must be rejected outright (not computed then rejected) - the whole point
    # is never materializing the enormous intermediate int.
    with pytest.raises(ValueError):
        safe_eval("9 ** 99999999999")


def test_safe_eval_rejects_result_exceeding_magnitude_cap():
    with pytest.raises(ValueError):
        safe_eval("10 ** 20")


def test_safe_eval_rejects_oversized_literal():
    with pytest.raises(ValueError):
        safe_eval("1e20 + 1")


def test_safe_eval_allows_results_within_magnitude_cap():
    assert safe_eval("2 ** 40") == 2**40


def test_safe_eval_rejects_bool_literals():
    # bool is a subclass of int in Python; "True"/"False" must not be treated as numeric literals.
    with pytest.raises(ValueError):
        safe_eval("True + 1")


def test_safe_eval_rejects_deeply_nested_expression_as_value_error():
    # A long chain of unary operators can blow the recursion limit during ast.parse itself, before
    # _eval_node ever runs. This must surface as ValueError (per the documented contract), not an
    # uncaught RecursionError.
    with pytest.raises(ValueError):
        safe_eval("-" * 3000 + "1")
