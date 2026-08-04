# run: PYTHONPATH=. python scripts/apply_ineffective_filter.py
"""
Sets aside problems where C fails AND B1 also fails, for BOTH models that
have data. These are NOT part of the headline experimental pool -- this is
a diagnostic set, for spot-checking whether the fact library is missing
something or whether fact-injection has a genuine limit on these problems.
If fact-injection has a genuine limit, then it could be used as an additional bonus 
if A is able to solve some of them.
"""
import json
from collections import defaultdict

PILOT_RESULTS = "results/gen_number_theory_pilot.jsonl"
FULL_PROBLEM_SET = "data/problem_sets/number_theory_L1-2.jsonl"
OUTPUT_PATH = "data/problem_sets/number_theory_L1-2_fact_ineffective.jsonl"

rows = [json.loads(l) for l in open(PILOT_RESULTS, encoding="utf-8")]

by_problem = defaultdict(dict)
for r in rows:
    by_problem[r["problem_id"]][(r["condition"], r["model"])] = r["correct"]

flagged_ids = set()
for pid, results in by_problem.items():
    for model in ["gemma4-e4b", "qwen3-1.7b"]:
        c = results.get(("C", model))
        b1 = results.get(("B1", model))
        if c is False and b1 is False:
            flagged_ids.add(pid)
            break

all_problems = [json.loads(l) for l in open(FULL_PROBLEM_SET, encoding="utf-8")]
flagged = [p for p in all_problems if p["problem_id"] in flagged_ids]

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for p in flagged:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"Wrote {len(flagged)} fact-ineffective problems -> {OUTPUT_PATH}")