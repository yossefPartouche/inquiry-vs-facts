import json
import os

PROMPTS_DIR = "prompts"

TEMPLATE_FILES = {
    "C": "C_zero_shot.txt",
    "A": "condition_A_inquiry.txt",
    "B1": "B1_bare_fact.txt",
    "B2": "B2_facts_no_combination.txt",
    "B3": "B3_near_analogous.txt",
    "B4": "B4_full_procedure.txt",
}

FACT_LIBRARY_PATH = os.path.join(PROMPTS_DIR, "fact_library.json")
B3_CONTENT_PATH = "data/b3_content/number_theory_L1-2_B3.jsonl"
B4_CONTENT_PATH = "data/b4_content/number_theory_L1-2_B4.jsonl"

_b3_cache = None
_b4_cache = None


class MissingContentError(Exception):
    """Raised when B3/B4 per-problem content hasn't been authored yet."""
    pass


def _load_template(condition):
    path = os.path.join(PROMPTS_DIR, TEMPLATE_FILES[condition])
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _facts_text(problem, condition):
    with open(FACT_LIBRARY_PATH, "r", encoding="utf-8") as f:
        library = json.load(f)
    key = "b1_facts" if condition == "B1" else "b2_facts"
    facts = library[problem["problem_type"]][key]
    return "\n".join(facts)  # bare lines, no bullets — B1/B2 render consistently


def _load_b3_index():
    global _b3_cache
    if _b3_cache is None:
        _b3_cache = {}
        with open(B3_CONTENT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                _b3_cache[row["problem_id"]] = row
    return _b3_cache


def _load_b4_index():
    global _b4_cache
    if _b4_cache is None:
        _b4_cache = {}
        with open(B4_CONTENT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                _b4_cache[row["problem_id"]] = row
    return _b4_cache


def build_prompt(problem, condition):
    if condition not in TEMPLATE_FILES:
        raise ValueError(f"Unknown condition: {condition!r}")

    template = _load_template(condition)

    if condition in ("B1", "B2"):
        template = template.replace("{FACTS}", _facts_text(problem, condition))

    elif condition == "B3":
        entry = _load_b3_index().get(problem["problem_id"])
        if entry is None:
            raise MissingContentError(
                f"no B3 content authored for problem_id {problem['problem_id']} yet"
            )
        template = template.replace("{ANALOGOUS_PROBLEM}", entry["analogous_problem"])
        template = template.replace("{ANALOGOUS_SOLUTION}", entry["analogous_solution"])

    elif condition == "B4":
        entry = _load_b4_index().get(problem["problem_id"])
        if entry is None:
            raise MissingContentError(
                f"no B4 content authored for problem_id {problem['problem_id']} yet"
            )
        template = template.replace("{PROCEDURE_STEPS}", entry["procedure_steps"])

    return template.replace("{PROBLEM}", problem["problem"])