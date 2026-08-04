# src/data_loader.py
"""
Loads MATH dataset problems, filtered to Number Theory, levels 1-2.
Assigns a stable, deterministic problem_id derived from the source
dataset's own unique_id field (never random, never insertion-order-based).
"""
from datasets import load_dataset

DATASET_NAME = "nlile/hendrycks-MATH-benchmark"
TARGET_SUBJECT = "Number Theory"
TARGET_LEVELS = (1, 2)


def load_number_theory_problems():
    """
    Returns a list of dicts, one per problem, each containing:
      problem_id, problem, gold, subject, level, source_unique_id
    (problem_type is added later by tag_problem_type — not here.)
    """
    ds = load_dataset(DATASET_NAME)

    rows = []
    for split_name in ("train", "test"):
        for record in ds[split_name]:
            if record["subject"] != TARGET_SUBJECT:
                continue
            if record["level"] not in TARGET_LEVELS:
                continue
            rows.append(record)

    # Sort by the dataset's own unique_id string. This is what makes
    # problem_id assignment reproducible run-to-run.
    rows.sort(key=lambda r: r["unique_id"])

    result = []
    counters = {1: 0, 2: 0}
    for record in rows:
        level = record["level"]
        counters[level] += 1
        problem_id = f"number_theory__{level}__{counters[level]:04d}"
        result.append({
            "problem_id": problem_id,
            "problem": record["problem"],
            "gold": record["answer"],
            "subject": "number_theory",
            "level": level,
            "source_unique_id": record["unique_id"],  # traceability, not required
        })

    return result
