"""
Validates data/problem_sets/number_theory_L1-2.jsonl against the loader's
contract: required fields, unique ids, valid levels, valid problem_type,
and reproducibility across repeated runs.
"""

from src.build_problem_set import build_and_write_problem_set
from src.problem_tagging import VALID_TYPES

REQUIRED_FIELDS = {"problem_id", "problem", "gold", "subject", "level", "problem_type"}


def test_all_rows_have_required_fields():
    rows = build_and_write_problem_set()
    for row in rows:
        missing = REQUIRED_FIELDS - row.keys()
        assert not missing, f"row {row.get('problem_id')} missing fields: {missing}"


def test_problem_id_is_unique():
    rows = build_and_write_problem_set()
    ids = [row["problem_id"] for row in rows]
    assert len(ids) == len(set(ids)), "duplicate problem_id found"


def test_level_is_1_or_2_only():
    rows = build_and_write_problem_set()
    for row in rows:
        assert row["level"] in (1, 2), f"row {row['problem_id']} has level {row['level']}"


def test_problem_type_is_known_and_not_blank():
    rows = build_and_write_problem_set()
    for row in rows:
        assert row["problem_type"], f"row {row['problem_id']} has blank problem_type"
        assert row["problem_type"] in VALID_TYPES, (
            f"row {row['problem_id']} has unknown problem_type: {row['problem_type']}"
        )


def test_reproducible_ids_across_runs():
    rows_first_run = build_and_write_problem_set()
    rows_second_run = build_and_write_problem_set()

    ids_first = [row["problem_id"] for row in rows_first_run]
    ids_second = [row["problem_id"] for row in rows_second_run]

    assert ids_first == ids_second, "problem_id assignment changed between runs"