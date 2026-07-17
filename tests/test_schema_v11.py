"""
Regression tests for the v1.1 schema change (parse_ok + status split).

Run either way, from the repo root:
    python tests/test_schema_v11.py      # prints PASS/FAIL lines, exits non-zero on failure
    pytest tests/test_schema_v11.py      # each check is a test_* function

Path-robust: resolves the repo root from this file's location, so it does not
depend on PYTHONPATH or the current working directory.
"""
import os
import sys
import json
import copy
import src.schema as S
from src.grader import grade, GoldParseError

# --- make `import schema` / `import grader` work no matter where we're invoked ---
# The modules live in src/ and import each other flatly (grader.py does
# `from schema import ...`), i.e. src/ is the source root on sys.path. Under
# pytest this is done by conftest.py; we repeat it here so the standalone
# `python tests/test_schema_v11.py` path (which does NOT load conftest) works too.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

FIXTURES = os.path.join(_HERE, "example_rows.jsonl")

BASE = dict(
    problem_id="number_theory__1__0007", subject="number_theory", level=1,
    condition="C", model="gemma4-e4b", seed=0, prompt_tokens=96,
    raw_output="...\\boxed{2}", gold="2",
)


def _row(**overrides):
    r = S.make_row(**BASE)
    r.update(overrides)
    return r


# ---------------------------------------------------------------------------
# 1. Requirement 4: a row written BEFORE grading (parse_ok is None) is VALID.
# ---------------------------------------------------------------------------
def test_pregrading_row_is_valid():
    r = S.make_row(**BASE)
    assert r["parse_ok"] is None
    assert r["correct"] is None
    assert r["parsed_answer"] is None
    assert r["schema_version"] == "1.1"
    S.validate_row(r)  # must not raise


# ---------------------------------------------------------------------------
# 2. parse_ok is DERIVED from status, in exactly one place.
# ---------------------------------------------------------------------------
def test_parse_ok_derivation():
    assert S.parse_ok_from_status("ok") is True
    assert S.parse_ok_from_status("multiple_found") is True
    assert S.parse_ok_from_status("grader_error") is True
    assert S.parse_ok_from_status("none_found") is False
    assert S.parse_ok_from_status("unparseable") is False
    assert S.parse_ok_from_status("empty_output") is False
    assert S.parse_ok_from_status("refusal") is False
    assert S.parse_ok_from_status(None) is None


def test_constants_stay_in_sync():
    # if this fails, someone added a status to one set but not the other,
    # which would silently skip the contradiction check for that status.
    assert set(S.PARSE_OK_BY_STATUS) == S.BOX_STATUSES


# ---------------------------------------------------------------------------
# 3. A fully graded, consistent row round-trips.
# ---------------------------------------------------------------------------
def test_graded_row_roundtrips():
    r = S.make_row(**BASE)
    g = S.make_grade_record(r["row_id"], "2", True, "ok", all_boxed_matches=["2"])
    r.update({k: v for k, v in g.items() if k != "row_id"})
    assert r["parse_ok"] is True
    S.validate_row(r)


# ---------------------------------------------------------------------------
# 4. Rejections: the guardrails that make parse_ok trustworthy.
# ---------------------------------------------------------------------------
def _assert_rejects(row, expect_substr):
    try:
        S.validate_row(row)
    except ValueError as e:
        assert expect_substr in str(e), f"wrong message: {e}"
        return
    raise AssertionError(f"expected rejection containing {expect_substr!r}, but row was accepted")


def test_reject_contradiction_none_found_true():
    _assert_rejects(_row(box_extraction_status="none_found", parse_ok=True), "contradicts")


def test_reject_contradiction_ok_false():
    _assert_rejects(_row(box_extraction_status="ok", parse_ok=False), "contradicts")


def test_reject_parse_ok_wrong_type():
    _assert_rejects(_row(parse_ok="true"), "parse_ok must be bool")


def test_reject_half_graded():
    _assert_rejects(_row(box_extraction_status="ok", parse_ok=None), "parse_ok is null")


def test_reject_invalid_status():
    _assert_rejects(_row(box_extraction_status="banana", parse_ok=False),
                    "box_extraction_status must be one of")


def test_reject_bad_all_boxed_matches():
    _assert_rejects(_row(all_boxed_matches=[1, 2]), "all_boxed_matches")


# ---------------------------------------------------------------------------
# 5. The confound, pinned. THIS is the 9.5% case: a fluent, possibly-correct
#    answer with no \boxed{} must be none_found / parse_ok False / correct False,
#    NOT silently graded as a wrong answer.
# ---------------------------------------------------------------------------
def test_confound_fluent_no_box():
    out = grade("Since 2^6 = 1 mod 7 and 100 = 6*16+4, the remainder is 2.", "2")
    assert out["box_extraction_status"] == "none_found"
    assert out["parse_ok"] is False
    assert out["correct"] is False        # policy 1: counts as incorrect


def test_multiple_found_last_box_wins():
    out = grade("first \\boxed{3}, actually \\boxed{2}", "2")
    assert out["box_extraction_status"] == "multiple_found"
    assert out["parse_ok"] is True
    assert out["correct"] is True


def test_empty_output_status():
    out = grade("", "2")
    assert out["box_extraction_status"] == "empty_output"
    assert out["parse_ok"] is False


def test_gold_bug_raises_not_silently_wrong():
    try:
        grade("\\boxed{2}", "")
    except GoldParseError:
        return
    raise AssertionError("empty gold must raise GoldParseError, not score as a wrong answer")


# ---------------------------------------------------------------------------
# 6. Any fixtures already migrated to v1.1 must validate. (v1.0 rows skipped:
#    they are the migration's INPUT, not its output.)
# ---------------------------------------------------------------------------
def test_migrated_fixtures_validate():
    if not os.path.exists(FIXTURES):
        return  # fixtures optional; don't fail CI if they're absent
    seen = 0
    for line in open(FIXTURES, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("schema_version") == "1.1":
            S.validate_row(row)
            seen += 1
    # no assertion on `seen`: file may legitimately still be pre-migration


# ---------------------------------------------------------------------------
# Standalone runner (no pytest needed).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS | {name}")
        except AssertionError as e:
            failures.append(name)
            print(f"FAIL | {name} -> {e}")
        except Exception as e:
            failures.append(name)
            print(f"ERR  | {name} -> {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        sys.exit(1)
    print(f"all {len(tests)} passed")