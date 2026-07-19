# run: PYTHONPATH=. python scripts/apply_selection_filter.py
"""
Freezes (stores) the problem set for the headline run: keep problems where C fails
but B1 rescues, for at least one model. This becomes the pool for ALL SIX
conditions from here on -- never regenerate or re-filter after this point,
or the paired design breaks.
"""
import json
from collections import defaultdict

PILOT_RESULTS = "results/gen_number_theory_pilot.jsonl"
FULL_PROBLEM_SET = "data/problem_sets/number_theory_L1-2.jsonl"
OUTPUT_PATH = "data/problem_sets/number_theory_L1-2_filtered.jsonl"

rows = [json.loads(l) for l in open(PILOT_RESULTS, encoding="utf-8")]

by_problem = defaultdict(dict)
for r in rows:
    by_problem[r["problem_id"]][(r["condition"], r["model"])] = r["correct"]

kept_ids = set()
for pid, results in by_problem.items():
    for model in ["gemma4-e4b", "qwen3-1.7b"]:
        c = results.get(("C", model))
        b1 = results.get(("B1", model))
        if c is False and b1 is True:
            kept_ids.add(pid)
            break

all_problems = [json.loads(l) for l in open(FULL_PROBLEM_SET, encoding="utf-8")]
filtered = [p for p in all_problems if p["problem_id"] in kept_ids]

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for p in filtered:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"Wrote {len(filtered)} filtered problems -> {OUTPUT_PATH}")