# run: PYTHONPATH=. python -m scripts.backfill_missing
"""
Finds (problem_id, condition, model) combinations missing from a results
file and re-runs ONLY those, appending the missing rows. Use after a run
that crashed or stopped early.
"""
import json
from collections import defaultdict

from src.runner import run, load_problem_set

RESULTS_PATH = "results/gen_number_theory_pilot.jsonl"
CONDITIONS = ("C", "B1")
MODEL = "qwen3-1.7b"

rows = [json.loads(l) for l in open(RESULTS_PATH, encoding="utf-8")]
present = {(r["problem_id"], r["condition"]) for r in rows if r["model"] == MODEL}

all_problems = load_problem_set()
missing_problems = [
    p for p in all_problems
    if any((p["problem_id"], c) not in present for c in CONDITIONS)
]

print(f"Backfilling {len(missing_problems)} problems for {MODEL}")

run(
    conditions=CONDITIONS,
    model_keys=(MODEL,),
    problems=missing_problems,
    max_tokens=1024,
    output_path=RESULTS_PATH,
)