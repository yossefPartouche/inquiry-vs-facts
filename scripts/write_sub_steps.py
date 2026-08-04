# scripts/write_sub_steps.py
import json
from scripts.parse_sub_steps import parse_sub_steps

path = "results/gen_number_theory_headline.jsonl"
rows = [json.loads(l) for l in open(path, encoding="utf-8")]

updated = 0
for r in rows:
    if r["condition"] == "A":
        steps = parse_sub_steps(r["raw_output"])
        r["sub_steps"] = steps if steps else None
        updated += 1

with open(path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Updated sub_steps for {updated} condition-A rows")