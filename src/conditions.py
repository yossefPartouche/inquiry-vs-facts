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

def build_prompt(problem, condition):
    if condition not in TEMPLATE_FILES:
        raise ValueError(f"Unknown condition: {condition!r}")

    if condition in ("B3", "B4"):
        raise MissingContentError(
            f"no B3/B4 content authored for problem_id {problem['problem_id']} yet"
        )

    template = _load_template(condition)

    if condition in ("B1", "B2"):
        template = template.replace("{FACTS}", _facts_text(problem, condition))

    return template.replace("{PROBLEM}", problem["problem"])