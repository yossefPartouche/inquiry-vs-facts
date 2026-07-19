# run: PYTHONPATH=. python scripts/grade_pilot.py
import json
import re
from src.grader import grade, GoldParseError

path = "results/gen_number_theory_pilot.jsonl"

def extract_dollar_math_fallback(raw_output):
    """
    Fallback for Gemma-style outputs that give the final answer in
    $$...$$ or $...$ display math instead of \boxed{}. Only used when
    \boxed{} extraction fails. Takes the LAST math block in the output,
    on the assumption the final answer is stated last (consistent with
    how \boxed{} extraction already takes the last box).
    """
    matches = re.findall(r"\$\$(.+?)\$\$|\$(.+?)\$", raw_output, re.DOTALL)
    if not matches:
        return None
    last = matches[-1]
    content = last[0] or last[1]
    return content.strip()

rows = [json.loads(l) for l in open(path, encoding="utf-8")]
graded = 0
gold_bugs = 0

for r in rows:
    try:
        g = grade(r["raw_output"], r["gold"])
    except GoldParseError as e:
        print(f"  !! GOLD BUG {r['row_id']}: {e}")
        gold_bugs += 1
        continue
    if g["box_extraction_status"] == "none_found":
            fallback_answer = extract_dollar_math_fallback(r["raw_output"])
            if fallback_answer is not None:
                # wrap it as if it were boxed, so it goes through the SAME
                # sympy-equivalence check -- don't hand-roll a separate comparison
                g = grade(f"\\boxed{{{fallback_answer}}}", r["gold"])
                g["box_extraction_status"] = "dollar_math_fallback"  # keep it distinguishable
                
    r.update(g)
    graded += 1

with open(path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Graded {graded}/{len(rows)} rows ({gold_bugs} gold-parse bugs)")


g = grade(r["raw_output"], r["gold"])
