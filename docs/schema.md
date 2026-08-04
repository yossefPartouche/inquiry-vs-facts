# Results Schema — EARLY DESIGN DRAFT (not the final implementation)

**This document describes an early planning proposal and does not reflect
the final schema.** The implemented system uses a single-file-per-run
design with grading fields written in-place (see `src/schema.py` for the
authoritative, current schema). Kept here for historical record of the
initial design discussion.

## Current Schema (implemented, see `src/schema.py`)

One JSON row per model call, written via `make_row()` + `append_jsonl()`.
Never hand-build a row.

**File organization:** one file per run, not per subject. Grading fields
are written into the same row (not a separate file), filled in a
**separate pass** after generation (`src/grading_pipeline.py`), so
re-grading never requires re-running a model.

results/
gen_number_theory_pilot.jsonl # C + B1 pilot, selection filter
gen_number_theory_headline.jsonl # all 6 conditions, frozen 110-problem pool
gen_number_theory_ablation.jsonl # A_corrupt (last-step)
gen_number_theory_ablation_all.jsonl # A_corrupt_all
gen_number_theory_ablation_swap.jsonl # A_swap

### Fields

| Field | Written by | Notes |
|---|---|---|
| `schema_version` | runner | current: see `SCHEMA_VERSION` in `src/schema.py` |
| `problem_id`, `subject`, `level` | runner | from the frozen problem set |
| `condition` | runner | `C, A, B1, B2, B3, B4, A_corrupt, A_corrupt_all, A_swap` |
| `model` | runner | `gemma4-e4b, qwen3-1.7b` |
| `seed`, `prompt_tokens`, `raw_output`, `finish_reason`, `gold` | runner | written at generation time; `raw_output` is never null |
| `parsed_answer`, `correct`, `box_extraction_status`, `parse_ok` | grader | null until graded; grader is a separate pass |
| `all_boxed_matches`, `grader_version`, `graded_at` | grader | audit trail |
| `sub_steps` | chain parser | condition A only; `[{step, sub_question, sub_answer}]`; null until parsed, `[]` if unparseable |
| `timestamp` | runner | ISO 8601, call time |

### `box_extraction_status` values (grader, in order of application)

`ok` → `none_found` → `dollar_math_fallback` → `final_is_fallback` →
`manual_review_correct` / `manual_review_incorrect` / `manual_review_incomplete`

The last three exist because a meaningful fraction of completions (up to
100% under `A_corrupt_all`) required human verdicting — models frequently
stated answers in free-form prose the automated fallbacks didn't catch.
See `analysis/headline_report.txt` for exact counts per run.

### Grading policy

Primary metric counts any row without a resolvable `correct=true` as
incorrect, including `A_corrupt_all`'s ungraded gaps and edge cases
(handled row-by-row, not silently excluded). `parse_ok` is reported
per-condition alongside accuracy, not folded into it silently.