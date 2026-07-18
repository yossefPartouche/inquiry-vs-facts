import pytest
from src.conditions import build_prompt, MissingContentError

MOD_EXP_PROBLEM = {
    "problem_id": "number_theory__1__0001",
    "problem": "What is 2^100 mod 7?",
    "gold": "2",
    "subject": "number_theory",
    "level": 1,
    "problem_type": "modular_exponentiation",
}

MOD_ARITH_PROBLEM = {
    "problem_id": "number_theory__1__0002",
    "problem": "What is 17 mod 5?",
    "gold": "2",
    "subject": "number_theory",
    "level": 1,
    "problem_type": "modular_arithmetic",
}


def test_C_contains_problem_text():
    prompt = build_prompt(MOD_EXP_PROBLEM, "C")
    assert MOD_EXP_PROBLEM["problem"] in prompt
    assert "{PROBLEM}" not in prompt


def test_A_contains_problem_text():
    prompt = build_prompt(MOD_EXP_PROBLEM, "A")
    assert MOD_EXP_PROBLEM["problem"] in prompt
    assert "{PROBLEM}" not in prompt


def test_B1_B2_facts_differ():
    b1 = build_prompt(MOD_EXP_PROBLEM, "B1")
    b2 = build_prompt(MOD_EXP_PROBLEM, "B2")
    assert "{FACTS}" not in b1 and "{FACTS}" not in b2
    assert "Fermat" in b1
    assert len(b2) > len(b1)


def test_B1_B2_modular_arithmetic():
    b1 = build_prompt(MOD_ARITH_PROBLEM, "B1")
    b2 = build_prompt(MOD_ARITH_PROBLEM, "B2")
    assert "congruent" in b1
    assert len(b2) > len(b1)


def test_B3_B4_raise_missing_content():
    for condition in ("B3", "B4"):
        with pytest.raises(MissingContentError) as exc:
            build_prompt(MOD_EXP_PROBLEM, condition)
        assert MOD_EXP_PROBLEM["problem_id"] in str(exc.value)


def test_unknown_condition_raises():
    with pytest.raises(ValueError):
        build_prompt(MOD_EXP_PROBLEM, "Z9")