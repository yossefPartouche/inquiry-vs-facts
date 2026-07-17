"""Test suite for src/grader.py

26 cases covering:
  - Core extraction & normalization (fractions, decimals, commas, units, nesting)
  - True negatives (wrong answers stay wrong)
  - Number Theory specifics (modular arithmetic, large powers, symbolic)
  - Robustness (whitespace, deeply nested, garbage after box, no box in gold)
  - Edge cases (empty box, unbalanced, no braces, unparseable input)

Run: pytest tests/test_grader.py -v
"""

import pytest
from src.grader import grade, extract_boxed, GoldParseError


# ============================================================================
# CORE BEHAVIOR (from original battery)
# ============================================================================

def test_extract_boxed_simple_int():
    """Simple integer in box."""
    result = grade(r"So \boxed{6}", "6")
    assert result["correct"] is True
    assert result["parse_ok"] is True


def test_extract_boxed_frac_vs_decimal():
    """Fraction and decimal are equivalent."""
    result = grade(r"\boxed{\frac{1}{2}}", "0.5")
    assert result["correct"] is True
    assert result["parse_ok"] is True


def test_extract_boxed_both_boxed():
    """Both model and gold have boxes."""
    result = grade(r"\boxed{\frac{1}{2}}", r"\boxed{\frac{1}{2}}")
    assert result["correct"] is True
    assert result["parse_ok"] is True


def test_extract_boxed_last_not_first():
    """Models restate. Last box is the answer."""
    result = grade(r"Maybe \boxed{5}, wait no, \boxed{6}", "6")
    assert result["correct"] is True
    assert result["parse_ok"] is True


def test_extract_boxed_comma_in_number_braced():
    """Commas in \\{,\\} are preserved and still parse."""
    result = grade(r"\boxed{1{,}000}", "1000")
    assert result["correct"] is True
    assert result["parse_ok"] is True


def test_extract_boxed_comma_in_number_plain():
    """Plain commas in numbers."""
    result = grade(r"\boxed{1,000}", "1000")
    assert result["correct"] is True
    assert result["parse_ok"] is True


def test_extract_boxed_units_stripped():
    """Units are normalized away by math_verify."""
    result = grade(r"\boxed{7 \text{ units}}", "7")
    assert result["correct"] is True
    assert result["parse_ok"] is True


def test_extract_boxed_left_right_nesting():
    """\\left/\\right and nested fractions."""
    result = grade(r"\boxed{\left(\frac{3}{4}\right)}", r"\frac{3}{4}")
    assert result["correct"] is True
    assert result["parse_ok"] is True


def test_grade_true_negative():
    """Wrong answer stays wrong."""
    result = grade(r"\boxed{6}", "5")
    assert result["correct"] is False
    assert result["parse_ok"] is True


def test_grade_zero():
    """Zero doesn't get eaten by falsy checks."""
    result = grade(r"\boxed{0}", "0")
    assert result["correct"] is True
    assert result["parse_ok"] is True


def test_grade_no_box_in_output():
    """Model output with no \\boxed{}."""
    result = grade("The answer is 6.", "6")
    assert result["correct"] is False
    assert result["parse_ok"] is False


def test_extract_boxed_unbalanced_braces():
    """Unbalanced \\boxed{ with no closing brace."""
    result = grade(r"\boxed{6", "6")
    assert result["correct"] is False
    assert result["parse_ok"] is False


def test_extract_boxed_no_braces():
    """\\boxed without braces (whitespace fallback)."""
    result = grade(r"\boxed 5", "5")
    assert result["correct"] is True
    assert result["parse_ok"] is True


def test_extract_boxed_unevaluated_expr():
    """Unevaluated expressions (sympy handles them)."""
    result = grade(r"\boxed{2^3}", "8")
    assert result["correct"] is True
    assert result["parse_ok"] is True


def test_extract_boxed_escaped_braces_earlier():
    """Escaped braces \\{ \\} earlier in output don't affect scan."""
    result = grade(r"Set \{1,2\}: \boxed{3}", "3")
    assert result["correct"] is True
    assert result["parse_ok"] is True


def test_extract_boxed_empty_box():
    """\\boxed{} is not an answer."""
    result = grade(r"\boxed{}", "3")
    assert result["correct"] is False
    assert result["parse_ok"] is False


# ============================================================================
# NUMBER THEORY SPECIFICS
# ============================================================================

def test_grade_modular_arithmetic_pmod():
    """Modular arithmetic: \\pmod notation."""
    result = grade(r"\boxed{1 \pmod{7}}", r"1 \pmod{7}")
    assert result["correct"] is True
    assert result["parse_ok"] is True


def test_grade_modular_arithmetic_mod():
    """Modular arithmetic: \\mod notation."""
    result = grade(r"\boxed{3 \mod 5}", r"3 \mod 5")
    assert result["correct"] is True
    assert result["parse_ok"] is True


def test_grade_large_unevaluated_power():
    """Large powers that shouldn't be computed (2^100 is huge)."""
    result = grade(r"\boxed{2^{100}}", r"2^{100}")
    assert result["correct"] is True
    assert result["parse_ok"] is True


def test_grade_negative_number():
    """Negative integers (appear in differences)."""
    result = grade(r"\boxed{-5}", "-5")
    assert result["correct"] is True
    assert result["parse_ok"] is True


def test_grade_mixed_expression_evaluates():
    """Mixed expression that sympy can evaluate: 2^10 - 1 = 1023."""
    result = grade(r"\boxed{2^{10} - 1}", "1023")
    assert result["correct"] is True
    assert result["parse_ok"] is True


# ============================================================================
# ROBUSTNESS
# ============================================================================

def test_extract_boxed_whitespace_padding():
    """Whitespace inside the box is stripped."""
    result = grade(r"\boxed{  6  }", "6")
    assert result["correct"] is True
    assert result["parse_ok"] is True


def test_extract_boxed_decimal_trailing_zeros():
    """Trailing zeros in decimals are normalized."""
    result = grade(r"\boxed{0.50}", "0.5")
    assert result["correct"] is True
    assert result["parse_ok"] is True


def test_extract_boxed_deeply_nested():
    """Deeply nested LaTeX (fraction of fraction)."""
    result = grade(r"\boxed{\frac{\frac{1}{2}}{3}}", r"\frac{1}{6}")
    assert result["correct"] is True
    assert result["parse_ok"] is True


def test_grade_gold_without_box():
    """Gold answer has no \\boxed{}, model output does."""
    result = grade(r"\boxed{42}", "42")
    assert result["correct"] is True
    assert result["parse_ok"] is True


def test_grade_symbolic_answer():
    """Symbolic (not numeric) answers, e.g., sqrt(2)."""
    result = grade(r"\boxed{\sqrt{2}}", r"\sqrt{2}")
    assert result["correct"] is True
    assert result["parse_ok"] is True


def test_extract_boxed_garbage_after():
    """Garbage after the box is ignored."""
    result = grade(r"\boxed{6}\n\nWait, let me reconsider...", "6")
    assert result["correct"] is True
    assert result["parse_ok"] is True


# ============================================================================
# GOLD VALIDATION (crash on malformed golds)
# ============================================================================

def test_gold_empty_string_raises():
    """Empty gold string raises GoldParseError (data bug)."""
    with pytest.raises(GoldParseError):
        grade(r"\boxed{6}", "")


def test_gold_none_raises():
    """None gold raises TypeError (data bug)."""
    with pytest.raises((GoldParseError, TypeError)):
        grade(r"\boxed{6}", None)


def test_gold_word_as_symbol_parses_as_product():
    """Word golds parse as symbolic products (e.g., 'six' → s*i*x).
    
    This is a limitation of math_verify: it extracts any identifier as a variable.
    We cannot catch semantic errors at the grader level.
    MATH golds are LaTeX/numeric, so this isn't a real problem in practice.
    If you need to validate golds, check them in your schema upstream before grading.
    """
    # 'six' parses as s*i*x (a symbolic product), so it doesn't raise GoldParseError
    result = grade(r"\boxed{6}", "six")
    assert result["correct"] is False  # 6 ≠ s*i*x
    assert result["parse_ok"] is True


# ============================================================================
# BOUNDARY CASES
# ============================================================================

def test_extract_boxed_brace_runaway_prevented():
    """Regression: \\boxed 5 followed by unrelated braces doesn't grab them."""
    # Previously: \\boxed 5, from the set {1,2} would extract '1,2'
    # Now: extracts '5'
    result = grade(r"\boxed 5, from the set {1,2}", "5")
    assert result["correct"] is True
    assert result["parse_ok"] is True
    assert result["parsed_answer"] == "5"


def test_extract_boxed_multiple_open_braces():
    """Multiple opening braces in a row after \\boxed."""
    result = grade(r"\boxed{{6}}", "6")
    assert result["correct"] is True
    assert result["parse_ok"] is True