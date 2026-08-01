"""
results_schema.py
Data-format contract for the inquiry-vs-facts NLP project.

One JSON row per model call, appended to results/gen_<subject>.jsonl via
make_row() + append_jsonl(). Never hand-build these dicts elsewhere --
import from here so both people validate against the same rules.

Grading (parsed_answer, correct, box_extraction_status, ...) and the
condition-A sub_steps chain are filled by SEPARATE passes, not by the
runner. This module only covers the generation-time row plus the helpers
needed to apply those two passes later. See results_schema.md for the
full spec and rationale.
"""
import json
import os
from datetime import datetime, timezone

SCHEMA_VERSION = "1.3.0"   # 1.1.0 -> ...; 1.3.0 -> ablation labels (A_corrupt, A_swap)
# ^ ADJUST if your real predecessor wasn't 1.3.0. Both tracks re-import on a bump,
#   same discipline as the parse_ok add. Adding condition labels IS a contract
#   change: a runner on an old copy would reject an ablation row as unknown.

# --- CONDITION VOCABULARY ----------------------------------------------------
# The substitution curve, IN ORDER. Single source of truth for the ladder;
# stats.py imports THIS tuple, so the headline plot can never grow a phantom
# rung from a label that isn't a ladder step.
LADDER_CONDITIONS = ("C", "B1", "B2", "B3", "B4")

# The transferred-inquiry condition: a MARKED POINT on the curve, not a rung.
INQUIRY_CONDITION = "A"

# The six conditions that run on EVERY problem (ladder + inquiry). This is the
# set the paired design is balanced over by default.
MAIN_CONDITIONS = set(LADDER_CONDITIONS) | {INQUIRY_CONDITION}

# Ablation-only labels (the novelty result). NOT ladder rungs. They run ONLY on
# the subset of problems that were ablated, and are analysed as a SEPARATE
# paired comparison (A vs A_corrupt, A vs A_swap) using the same difference-CI
# machinery -- never placed on the ladder.
ABLATION_CONDITIONS = ("A_corrupt", "A_corrupt_all", "A_swap")

CONDITIONS = MAIN_CONDITIONS | set(ABLATION_CONDITIONS)

RUNG_LABELS = {
    "B1": "bare_fact",
    "B2": "facts_no_combination",
    "B3": "near_analogous_worked_example",
    "B4": "full_procedure_answer_withheld",
}

MODELS = {"gemma4-e4b", "qwen3-1.7b"}

SUBJECTS = {"number_theory", "counting_probability", "algebra"}
FINISH_REASONS = {"ok", "empty_output", "api_error", "truncated"}

# CHANGED (v1.1): added "unparseable" and "grader_error".
#   none_found     -- no \boxed{} anywhere in the output
#   unparseable    -- a \boxed{} IS present, but sympy could not parse its payload
# These two were previously collapsed into a single parse failure. They are
# different failures (format drift vs. malformed math) and condition A is
# expected to differ from C on BOTH, so they must be counted separately.
#   grader_error   -- extraction succeeded; the equivalence check itself blew up
#                     (sympy timeout/crash). Rare; if it isn't rare, that's a bug.

BOX_STATUSES = { "ok", "none_found", "multiple_found", "unparseable", "refusal", "empty_output", "grader_error"}
# NEW (v1.1): parse_ok is DERIVED from box_extraction_status -- never set it
# independently. Single source of truth: add a status above, add it here, done.
#
# "did the grader get a usable answer out of the model?"  NOT "was it right?"
#   multiple_found -> True : the last-\boxed rule still yields one usable answer.
#   grader_error   -> True : we DID extract an answer; the comparison failed.
#                            (It scores incorrect, but it is not a PARSE failure.)
# Both are judgement calls. Report their rates separately so nobody has to
# take our word for it.
PARSE_OK_BY_STATUS = {
    "ok":             True,
    "multiple_found": True,
    "grader_error":   True,
    "none_found":     False,
    "unparseable":    False,
    "empty_output":   False,
    "refusal":        False,
}

def parse_ok_from_status(box_extraction_status):
    """NEW (v1.1). The ONLY place parse_ok is computed. Grader pass calls this."""
    if box_extraction_status is None:
        return None                       # not graded yet
    if box_extraction_status not in PARSE_OK_BY_STATUS:
        raise ValueError(f"unknown box_extraction_status: {box_extraction_status!r}")
    return PARSE_OK_BY_STATUS[box_extraction_status]

def make_row_id(problem_id, condition, model, seed):
    return f"{problem_id}|{condition}|{model}|seed{seed}"


def problem_id(subject, level, index):
    """The ONLY way problem_id should be constructed. Call this from the
    problem-loading code with the source dataset index, and reuse the same
    (subject, level, index) for all 6 conditions x both models. Deriving
    the id by construction -- instead of trusting each caller to pass a
    matching string -- is what keeps the paired design from silently
    breaking."""
    if not isinstance(level, int) or not (1 <= level <= 5):
        raise ValueError("level must be an int 1-5")
    if not isinstance(index, int) or index < 0:
        raise ValueError("index must be a non-negative int")
    return f"{subject}__{level}__{index:04d}"


def make_row(
    *,
    problem_id,
    subject,
    level,
    condition,
    model,
    seed,
    prompt_tokens,
    raw_output,
    gold,
    finish_reason="ok",
    timestamp=None,
):
    """Build one generation-time row. Call this right after a model call
    returns, then pass the result to append_jsonl().

    Grading and sub_steps fields are deliberately left null here -- they
    get filled in later by apply_grade() / apply_substeps(), run over the
    file as a separate pass (see results_schema.md, section 3-4)."""

    rung_label = RUNG_LABELS.get(condition)  # None for C and A, fixed for B1-B4

    row = {
        "schema_version": SCHEMA_VERSION,
        "row_id": make_row_id(problem_id, condition, model, seed),
        "problem_id": problem_id,
        "subject": subject,
        "level": level,
        "condition": condition,
        "rung_label": rung_label,
        "model": model,
        "seed": seed,
        "prompt_tokens": prompt_tokens,
        "raw_output": raw_output,
        "finish_reason": finish_reason,
        "gold": gold,
        # grading fields: null until the grader pass runs
        "parsed_answer": None,
        "correct": None,
        "box_extraction_status": None,
        "parse_ok": None,
        "all_boxed_matches": None,
        "grader_version": None,
        "graded_at": None,
        # condition-A structured chain: null until the substep parser runs
        "sub_steps": None,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
    }
    validate_row(row)
    return row


def validate_row(row):
    """Raises ValueError with every problem found (not just the first)."""
    errors = []

    def require(cond, msg):
        if not cond:
            errors.append(msg)

    require(row.get("schema_version"), "schema_version is required")
    require(isinstance(row.get("problem_id"), str) and row["problem_id"], "problem_id must be a non-empty string")
    require(row.get("subject") in SUBJECTS, f"subject must be one of {SUBJECTS}")
    require(isinstance(row.get("level"), int) and 1 <= row.get("level", -1) <= 5, "level must be an int 1-5")
    require(row.get("condition") in CONDITIONS, f"condition must be one of {CONDITIONS}")

    condition = row.get("condition")
    rung = row.get("rung_label")
    if condition in RUNG_LABELS:
        require(rung == RUNG_LABELS[condition], f"rung_label must be '{RUNG_LABELS.get(condition)}' for condition {condition}")
    else:
        require(rung is None, "rung_label must be null for conditions C and A")

    require(row.get("model") in MODELS, f"model must be one of {MODELS}")
    require(isinstance(row.get("seed"), int), "seed must be an int")
    require(isinstance(row.get("prompt_tokens"), int) and row.get("prompt_tokens", -1) >= 0,
            "prompt_tokens must be a non-negative int")
    require(isinstance(row.get("raw_output"), str), "raw_output must be a string ('' for empty output, never null)")
    require(row.get("finish_reason") in FINISH_REASONS, f"finish_reason must be one of {FINISH_REASONS}")
    require(isinstance(row.get("gold"), str) and row.get("gold") != "", "gold must be a non-empty string")
    require(isinstance(row.get("row_id"), str) and row["row_id"], "row_id is required")
    require(isinstance(row.get("timestamp"), str) and row["timestamp"], "timestamp is required")

    if row.get("correct") is not None:
        require(isinstance(row["correct"], bool), "correct must be bool or null")
    status = row.get("box_extraction_status")
    if  status is not None:
        require(status in BOX_STATUSES,
                f"box_extraction_status must be one of {BOX_STATUSES} or null")
        
    parse_ok = row.get("parse_ok")
    if parse_ok is not None:
        require(isinstance(parse_ok, bool), "parse_ok must be bool or null")
    
    if parse_ok is not None and status is not None and status in PARSE_OK_BY_STATUS:
        require(parse_ok == PARSE_OK_BY_STATUS[status],
                f"parse_ok={parse_ok} contradicts box_extraction_status='{status}' "
                f"(expected {PARSE_OK_BY_STATUS[status]}); parse_ok is derived, do not set it by hand")
    if status is not None:
        require(parse_ok is not None, "graded row has box_extraction_status but parse_ok is null")
    
    if row.get("all_boxed_matches") is not None:
        require(isinstance(row["all_boxed_matches"], list)
                and all(isinstance(x, str) for x in row["all_boxed_matches"]),
                "all_boxed_matches must be a list of strings or null")
        
    sub_steps = row.get("sub_steps")
    if sub_steps is not None:
        require(condition == "A", "sub_steps must be null for non-A conditions")
        require(isinstance(sub_steps, list), "sub_steps must be a list (or null)")
        for i, step in enumerate(sub_steps):
            require(
                isinstance(step, dict) and {"step", "sub_question", "sub_answer"} <= step.keys(),
                f"sub_steps[{i}] must be an object with step/sub_question/sub_answer",
            )

    if errors:
        raise ValueError("Invalid row:\n  " + "\n  ".join(errors))
    return True


def append_jsonl(row, path):
    """Append one row to a generation JSONL file. This file is append-only:
    never rewrite or delete past rows here. Corrections belong in a fresh
    rerun row (new timestamp) or in the separate grades_/substeps_ files."""
    validate_row(row)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def gen_path(subject, results_dir="results"):
    return os.path.join(results_dir, f"gen_{subject}.jsonl")


def grades_path(subject, results_dir="results"):
    return os.path.join(results_dir, f"grades_{subject}.jsonl")


def substeps_path(subject, results_dir="results"):
    return os.path.join(results_dir, f"substeps_{subject}.jsonl")


# --- Helpers for the (separate) grading and sub-step-parsing passes ---
# These illustrate the expected shape; the actual grading/parsing logic
# (sympy equivalence, boxed-extraction, chain segmentation) lives in
# Track M's / Track X's own modules and is out of scope for this file.

def make_grade_record(row_id, parsed_answer, correct, box_extraction_status,
                      all_boxed_matches=None, grader_version="grader_v2", graded_at=None):
    """CHANGED (v1.1): now emits parse_ok, DERIVED from box_extraction_status.
    There is deliberately no parse_ok kwarg -- it cannot be passed in, so it
    cannot disagree with the status."""
    if box_extraction_status not in BOX_STATUSES:
        raise ValueError(f"box_extraction_status must be one of {BOX_STATUSES}")
    return {
        "row_id": row_id,
        "parsed_answer": parsed_answer,
        "correct": correct,
        "box_extraction_status": box_extraction_status,
        "parse_ok": parse_ok_from_status(box_extraction_status),   # NEW (v1.1)
        "all_boxed_matches": all_boxed_matches,
        "grader_version": grader_version,
        "graded_at": graded_at or datetime.now(timezone.utc).isoformat(),
    }


def make_substeps_record(row_id, sub_steps):
    """sub_steps: [] if parsed but no structure found, non-empty list if parsed ok."""
    if not isinstance(sub_steps, list):
        raise ValueError("sub_steps must be a list ([] if none found)")
    for i, step in enumerate(sub_steps):
        if not (isinstance(step, dict) and {"step", "sub_question", "sub_answer"} <= step.keys()):
            raise ValueError(f"sub_steps[{i}] must have step/sub_question/sub_answer")
    return {"row_id": row_id, "sub_steps": sub_steps}


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def join_on_row_id(gen_rows, grade_rows=None, substep_rows=None):
    """Merge gen_<subject>.jsonl with grades_<subject>.jsonl and
    substeps_<subject>.jsonl on row_id. Later duplicate row_ids in gen_rows
    (reruns) win -- 'last write wins'."""
    merged = {}
    for row in gen_rows:
        merged[row["row_id"]] = dict(row)  # last one wins on duplicate row_id

    for grade_rows_src in (grade_rows or []):
        rid = grade_rows_src["row_id"]
        if rid in merged:
            merged[rid].update({k: v for k, v in grade_rows_src.items() if k != "row_id"})

    for sub_row in (substep_rows or []):
        rid = sub_row["row_id"]
        if rid in merged:
            merged[rid]["sub_steps"] = sub_row["sub_steps"]

    return list(merged.values())


def assert_paired_completeness(rows, expected_conditions=None, expected_models=MODELS):
    """Run before any stats: group by problem_id and confirm every problem
    has exactly one row per (condition, model). Raises AssertionError with
    the specific gaps/dupes found -- catches a broken paired design before
    it corrupts a CI or effect size, instead of after.

    DEFAULT (expected_conditions=None) is the six MAIN_CONDITIONS (ladder + A),
    NOT all of CONDITIONS. The ablation labels (A_corrupt, A_swap) run only on
    the ablated subset, so demanding them on every problem would false-raise on
    a valid run. To check the ablation subset, call with
    expected_conditions=ABLATION_CONDITIONS on the ablated rows only."""
    from collections import defaultdict

    if expected_conditions is None:
        expected_conditions = MAIN_CONDITIONS

    by_problem = defaultdict(set)
    for row in rows:
        by_problem[row["problem_id"]].add((row["condition"], row["model"]))

    expected = {(c, m) for c in expected_conditions for m in expected_models}
    problems_bad = {}
    for pid, present in by_problem.items():
        missing = expected - present
        extra = present - expected
        if missing or extra:
            problems_bad[pid] = {"missing": sorted(missing), "extra": sorted(extra)}

    if problems_bad:
        raise AssertionError(
            f"Paired design broken for {len(problems_bad)} problem(s): {problems_bad}"
        )
    return True


if __name__ == "__main__":
    # smoke test
    row = make_row(
        problem_id="number_theory__1__0007",
        subject="number_theory",
        level=1,
        condition="B3",
        model="gemma4-e4b",
        seed=0,
        prompt_tokens=812,
        raw_output="...\\boxed{2}",
        gold="2",
    )
    append_jsonl(row, "results/gen_number_theory.jsonl")
    print("wrote:", row["row_id"])