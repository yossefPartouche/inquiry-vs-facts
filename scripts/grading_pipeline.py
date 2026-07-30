# src/grading_pipeline.py
# run: PYTHONPATH=. python -m src.grading_pipeline results/gen_number_theory_ablation.jsonl
"""
The single, canonical three-pass grading pipeline, used across every
results file in this project (pilot, headline, ablation). Consolidates
what was previously reimplemented separately in grade_pilot.py,
apply_dollar_fallback.py, apply_final_is_fallback.py, grade_ablation.py.
"""
import json
import re
from src.grader import grade, GoldParseError


def extract_dollar_math_fallback(raw_output):
    matches = re.findall(r"\$\$(.+?)\$\$|\$(.+?)\$", raw_output, re.DOTALL)
    if not matches:
        return None
    last = matches[-1]
    return (last[0] or last[1]).strip()


def extract_final_is_answer(raw_output):
    matches = re.findall(
        r"\bis\s+[\*'\"]{0,2}([\-\d./\\{}^a-zA-Z\s]+?)[\*'\"]{0,2}\.\s*(?:$|\n)",
        raw_output,
    )
    if not matches:
        return None
    return matches[-1].strip()


def grade_with_fallbacks(path):
    """Grades every row in `path` in place: primary grade, then dollar-math
    fallback, then free-form 'is X.' fallback, in that order. Writes the
    file back with grading fields populated. Returns (n_correct, n_total)."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]

    for r in rows:
        try:
            g = grade(r["raw_output"], r["gold"])
            r.update(g)
        except GoldParseError:
            pass

    for r in rows:
        if r.get("box_extraction_status") == "none_found":
            fb = extract_dollar_math_fallback(r["raw_output"])
            if fb is not None:
                try:
                    g = grade(f"\\boxed{{{fb}}}", r["gold"])
                    g["box_extraction_status"] = "dollar_math_fallback"
                    r.update(g)
                except GoldParseError:
                    pass

    for r in rows:
        if not r.get("correct"):
            candidate = extract_final_is_answer(r["raw_output"])
            if candidate is not None:
                try:
                    g = grade(f"\\boxed{{{candidate}}}", r["gold"])
                    if g.get("box_extraction_status") not in ("none_found", None) and g.get("correct"):
                        g["box_extraction_status"] = "final_is_fallback"
                        r.update(g)
                except GoldParseError:
                    pass

    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    correct = sum(1 for r in rows if r.get("correct"))
    return correct, len(rows)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "results/gen_number_theory_headline.jsonl"
    c, n = grade_with_fallbacks(path)
    print(f"{c}/{n} correct after grading + all fallbacks -> {path}")