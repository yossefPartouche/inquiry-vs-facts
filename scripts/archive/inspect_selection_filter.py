# run: PYTHONPATH=. python scripts/inspect_selection_filter.py
import json
from collections import defaultdict, Counter

rows = [json.loads(l) for l in open("results/gen_number_theory_pilot.jsonl")]

by_problem = defaultdict(dict)
for r in rows:
    by_problem[r["problem_id"]][(r["condition"], r["model"])] = r

for model in ["gemma4-e4b", "qwen3-1.7b"]:
    outcomes = Counter()
    for pid, results in by_problem.items():
        c_row = results.get(("C", model))
        b1_row = results.get(("B1", model))
        if c_row is None or b1_row is None:
            outcomes["missing_data"] += 1
            continue
        c, b1 = c_row["correct"], b1_row["correct"]
        if c is True:
            outcomes["C_already_correct (excluded: no headroom)"] += 1
        elif c is False and b1 is True:
            outcomes["C_fails_B1_rescues (KEPT)"] += 1
        elif c is False and b1 is False:
            outcomes["C_fails_B1_also_fails (fact didn't help)"] += 1
    print(f"--- {model} ---")
    for k, v in outcomes.most_common():
        print(f"  {k}: {v}")
    print()