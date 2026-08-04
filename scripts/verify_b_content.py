# scripts/verify_b_content.py
"""
Verifies authored B3/B4 content before it's trusted in the pipeline:
- checks the final answer doesn't leak into the procedure/example text
  (whole-number match, not naive substring -- avoids false positives like
  "4" matching inside "1445")
- checks every frozen problem has an entry, and no stray extra entries

Run for B4:
    PYTHONPATH=. python scripts/verify_b_content.py data/b4_content/number_theory_L1-2_B4.jsonl

Run for B3 (different field names):
    PYTHONPATH=. python scripts/verify_b_content.py data/b3_content/number_theory_L1-2_B3.jsonl --answer-field analogous_final_answer --text-field analogous_solution

KNOWN LIMITATION: whole-number matching still cannot catch non-literal leaks
(spelled-out numbers, implied answers, non-canonical fraction formats, an
answer restated in different notation). This is a first-pass net, not a
replacement for manually reading the content.
"""
import json
import re
import sys
import argparse

FROZEN_PATH = "data/problem_sets/number_theory_L1-2_filtered.jsonl"


def answer_leaks(answer, text):
    """True only if `answer` appears as a standalone number/token in `text`,
    not as a substring of a larger number (e.g. '4' inside '1445')."""
    answer = str(answer).strip()
    if not answer:
        return False
    pattern = r'(?<!\d)' + re.escape(answer) + r'(?!\d)'
    return re.search(pattern, text) is not None


def verify(content_path, answer_field="final_answer", text_field="procedure_steps"):
    rows = [json.loads(l) for l in open(content_path, encoding="utf-8")]

    issues = []
    for r in rows:
        answer = str(r.get(answer_field, "")).strip()
        text = r.get(text_field, "")

        if answer and answer_leaks(answer, text):
            issues.append((r["problem_id"], f"LEAK: {answer_field} appears in {text_field}"))
        if not text.strip():
            issues.append((r["problem_id"], f"EMPTY {text_field}"))
        if r.get("needs_review") or r.get("need_review"):
            issues.append((r["problem_id"], "FLAGGED for review"))

    print(f"{len(rows)} entries checked, {len(issues)} issues found")
    for pid, msg in issues:
        print(f"  {pid}: {msg}")

    frozen = {json.loads(l)["problem_id"] for l in open(FROZEN_PATH, encoding="utf-8")}
    content_ids = {r["problem_id"] for r in rows}
    missing = frozen - content_ids
    extra = content_ids - frozen

    print(f"\nmissing from {content_path}: {missing or 'none'}")
    print(f"extra (not in frozen set): {extra or 'none'}")

    return issues, missing, extra


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="data/b4_content/number_theory_L1-2_B4.jsonl")
    parser.add_argument("--answer-field", default="final_answer")
    parser.add_argument("--text-field", default="procedure_steps")
    args = parser.parse_args()

    verify(args.path, answer_field=args.answer_field, text_field=args.text_field)