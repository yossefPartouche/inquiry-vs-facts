# scripts/inspect_skipped_ablation.py
import json
from scripts.ablation_corrupt import corrupt_numeric_value

rows = [json.loads(l) for l in open("results/gen_number_theory_headline.jsonl")]
pool = [r for r in rows if r["condition"] == "A" and r["model"] == "gemma4-e4b" and r.get("correct")]

skipped = []
for r in pool:
    last_answer = r["sub_steps"][-1]["sub_answer"]
    if corrupt_numeric_value(last_answer) is None:
        skipped.append(r)

print(f"{len(skipped)} skipped")
for r in skipped:
    print(f"--- {r['problem_id']} ---")
    print(r["sub_steps"][-1]["sub_answer"])
    print()