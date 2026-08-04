# scripts/manual_correct_ablation.py
import json

# fill in the problem_ids you personally verified as correct but the
# automated grader missed
MANUALLY_CORRECT = [
    # "number_theory__2__00XX",
    # ...
]

path = "results/gen_number_theory_ablation.jsonl"
rows = [json.loads(l) for l in open(path, encoding="utf-8")]

fixed = 0
for r in rows:
    if r["problem_id"] in MANUALLY_CORRECT and not r.get("correct"):
        r["correct"] = True
        r["box_extraction_status"] = "manual_review_correct"
        fixed += 1

with open(path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Manually corrected {fixed} rows")