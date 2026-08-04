"""
Ties the data_loader.py and problem_tagging.py together and writes the frozen problem set to
data/problem_sets/number_theory_L1-2.jsonl (one JSON object per line).
"""
import json
import os

from src.data_loader import load_number_theory_problems
from src.problem_tagging import tag_problem_type

OUTPUT_PATH = "data/problem_sets/number_theory_L1-2.jsonl"


def build_and_write_problem_set(output_path=OUTPUT_PATH):
    """
    Loads + filters problems, tags each with a problem_type, and writes
    them as JSON Lines. Returns the list of rows written (handy for tests).
    """
    problems = load_number_theory_problems()

    for problem in problems:
        problem["problem_type"] = tag_problem_type(problem["problem"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for problem in problems:
            f.write(json.dumps(problem) + "\n")

    return problems


if __name__ == "__main__":
    rows = build_and_write_problem_set()
    print(f"Wrote {len(rows)} problems to {OUTPUT_PATH}")