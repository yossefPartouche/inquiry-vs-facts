# Results Data Format — Finalized Schema

One JSON row per model call. Three JSONL files per subject, joined on `row_id`:

- `results/gen_<subject>.jsonl` — written by the runner, at generation time. **Append-only, never edited.**
- `results/grades_<subject>.jsonl` — written by the grader, in a separate pass. **Fully regenerated each time you re-grade.**
- `results/substeps_<subject>.jsonl` — written by the chain parser (condition A only), also fully regenerated on re-parse.

Analysis code loads all three and joins on `row_id`.

## 1. Field-by-field spec

| Field | Type | Nullable | Written by | Meaning | Example |
|---|---|---|---|---|---|
| `schema_version` | string | no | runner | schema version tag | `"1.0"` |
| `row_id` | string | no | runner | deterministic join key: `problem_id\|condition\|model\|seed` | `"number_theory__1__0007\|B3\|gemma4-e4b\|seed0"` |
| `problem_id` | string | no | runner | stable id, identical across all 6 conditions × both models for the same problem | `"number_theory__1__0007"` |
| `subject` | string enum | no | runner | see §2 | `"number_theory"` |
| `level` | int (1–5) | no | runner | source dataset difficulty level | `1` |
| `condition` | string enum | no | runner | experimental arm | `"B3"` |
| `rung_label` | string enum | **yes** — null unless condition ∈ {B1–B4} | runner | human-readable description of what's injected | `"near_analogous_worked_example"` |
| `model` | string enum | no | runner | see §2 | `"gemma4-e4b"` |
| `seed` | int | no | runner | decode seed (log it even under greedy — some backends still consume it and it costs nothing) | `0` |
| `prompt_tokens` | int | no | runner | tokens in the full prompt sent to the model | `812` |
| `raw_output` | string | no (`""` allowed, not null) | runner | full raw text returned by the model | `"...\\boxed{2}"` |
| `finish_reason` | string enum | no | runner | mechanical outcome of the API call | `"ok"` |
| `gold` | string | no | runner | ground-truth answer, known before the call | `"2"` |
| `parsed_answer` | string | yes | **grader** | extracted final answer | `"2"` |
| `correct` | bool | yes | **grader** | grading verdict | `true` |
| `box_extraction_status` | string enum | yes | **grader** | diagnostic on `\boxed{}` extraction | `"ok"` |
| `all_boxed_matches` | array[string] | yes | **grader** | every `\boxed{}` match found (audit trail when >1) | `["8","2"]` |
| `grader_version` | string | yes | **grader** | tag of the grading logic used — lets you tell which pass produced a verdict | `"grader_v1"` |
| `graded_at` | string (ISO 8601) | yes | **grader** | when grading ran | `"2026-07-15T10:22:00Z"` |
| `sub_steps` | array[object] or `[]` | yes — null until parsed; non-null only for condition A | **substep parser** | structured sub-question/sub-answer chain, for corruption/swap ablations | see §4 |
| `timestamp` | string (ISO 8601) | no | runner | when the model call was made | `"2026-07-14T09:03:11Z"` |

`raw_output` is the only field that must never be null — every edge case (empty generation, refusal, no box found) still produces a row with a real (possibly empty) string there. Everything else that can legitimately fail to exist is null, never a sentinel string.

## 2. Enumerated values

- **`condition`**: `C`, `A`, `B1`, `B2`, `B3`, `B4`
- **`rung_label`** (fixed 1:1 with condition, only for B-rungs):
  - `B1` → `bare_fact`
  - `B2` → `facts_no_combination`
  - `B3` → `near_analogous_worked_example`
  - `B4` → `full_procedure_answer_withheld`
- **`model`**: `gemma4-e4b`, `qwen3-1.7b`
- **`subject`**: `number_theory` (Tier 1, active now), `counting_probability`, `algebra` (Tiers 2–3, reserved so the schema doesn't need to change if you get to them)
- **`finish_reason`** (runner-observed, mechanical): `ok`, `empty_output`, `api_error`, `truncated`
- **`box_extraction_status`** (grader-observed, semantic): `ok`, `none_found`, `multiple_found`, `refusal`, `empty_output`

Splitting `finish_reason` (what the API call did) from `box_extraction_status` (what the grader found in the text) matters because they can disagree — e.g. `finish_reason="ok"` but the model still refused, or rambled with two boxed answers.

## 3. Grading: separate pass (recommended — matches your lean)

Leave `parsed_answer`/`correct`/`box_extraction_status` **null at generation time**, filled by a distinct grader pass that reads `gen_<subject>.jsonl` and writes `grades_<subject>.jsonl`. Reasons:

1. **Re-grading is free, re-running models isn't.** Your grader (sympy equivalence, `\boxed{}` normalizers) will have bugs and edge cases you find in week 2–3. A separate pass means fixing them costs a script run, not a model rerun.
2. **It matches your own Track split.** Your plan already assigns generation to Track X and grading/stats to Track M. A separate grader pass makes that division literal in the data — Track M can iterate on `grades_*.jsonl` all week without touching Track X's files.
3. **`gen_*.jsonl` stays a true append-only log.** If grading fields lived in the same row as the runner writes it, "re-grading" would mean mutating historical rows in place — risky for a file two people are both appending to. Keeping generation immutable and grading regenerable avoids merge conflicts and lets you diff grader output run-to-run.
4. **`grader_version` gives you an audit trail** for which grading logic produced which verdict — useful when your report needs to say "we re-graded after fixing the fraction normalizer."

## 4. Condition A's chain: raw_output + a separate structured field

Keep `raw_output` as the sole authoritative record of what the model actually said — never derive ablations from anything else. Additionally, add `sub_steps` (nullable, condition-A-only) populated by a **separate parser pass** (same append/regenerate split as grading), not by the runner:

```json
"sub_steps": [
  {"step": 1, "sub_question": "...", "sub_answer": "..."},
  {"step": 2, "sub_question": "...", "sub_answer": "..."}
]
```

- `null` = not yet parsed.
- `[]` = parsed, but no clean sub-Q/sub-A structure was detected (log it, don't silently drop the row).
- non-empty list = ready for chain-corruption / question-swap ablations to manipulate directly, without re-parsing `raw_output` text every time.

This is the same "cheap to redo, expensive to regenerate from the model" logic as grading — if your parsing heuristic improves in week 3, you rerun the parser, not the models.

## 5. File organization

```
results/
  gen_number_theory.jsonl        # runner writes here, append-only
  grades_number_theory.jsonl     # grader writes here, full overwrite each grading run
  substeps_number_theory.jsonl   # substep parser writes here, condition A only, full overwrite
```

One file per subject (not per run) — reruns just append more rows with a fresh `timestamp`; `row_id` collisions (identical problem/condition/model/seed rerun) are resolved by "last write wins" at load time, so a noisy rerun doesn't require a new filename. If you reach Tier 2/3, add `gen_counting_probability.jsonl` etc. — same shape, no schema change needed. No need to shard by date or by person; git history + `timestamp` already gives you that.

## 6. Edge cases

| Case | `finish_reason` | `raw_output` | `box_extraction_status` | `parsed_answer` | `correct` |
|---|---|---|---|---|---|
| Normal, one `\boxed{}` | `ok` | full text | `ok` | extracted value | computed |
| No `\boxed{}` anywhere | `ok` | full text | `none_found` | `null` | `false` |
| Refusal (e.g. "I can't help with that") | `ok` | refusal text | `refusal` | `null` | `false` |
| Multiple `\boxed{}` | `ok` | full text | `multiple_found` | **last** match (convention: model's final answer supersedes earlier work) — all matches logged in `all_boxed_matches` | computed on the last match |
| Empty generation (API returned nothing) | `empty_output` | `""` | `empty_output` | `null` | `false` |
| API/runtime error | `api_error` | `""` or partial | `null` (grader never sees a usable row — flag for rerun, exclude from analysis) | `null` | `null` |

Anything the grader can't extract a real answer from is scored `correct=false` (a non-attempt fails), except `api_error`, which should be excluded from analysis and rerun rather than scored — that's the one case that isn't the model's fault.

## 7. `schema_version`

Yes, add it. One month, two people, a stretch phase (LoRA, extra subjects) that may add fields later — a version tag costs one line and lets loader scripts assert the shape they expect instead of failing silently on a field that got renamed mid-project.

## README note (paste into repo README)

> Every model call produces exactly one row, appended to `results/gen_<subject>.jsonl` via `make_row()` + `append_jsonl()` in `results_schema.py` — never hand-build the JSON. Rows are keyed by `row_id` (`problem_id|condition|model|seed`), which is what lets the paired analysis join all six conditions back to the same problem. Grading and condition-A chain-parsing are separate, rerunnable passes: they read `gen_<subject>.jsonl` and write `grades_<subject>.jsonl` / `substeps_<subject>.jsonl` respectively, fully regenerated each time (not appended), so fixing a grader bug or a chain-parsing heuristic never requires rerunning a model. `raw_output` is the only field that's never null — it's the ground truth everything else derives from.