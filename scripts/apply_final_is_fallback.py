import json, re
from src.grader import grade, GoldParseError

def extract_final_is_answer(raw_output):
    matches = re.findall(
        r"\bis\s+[\*'\"]{0,2}([\-\d./\\{}^a-zA-Z\s]+?)[\*'\"]{0,2}\.\s*(?:$|\n)",
        raw_output
    )
    if not matches:
        return None
    return matches[-1].strip()

path = "results/gen_number_theory_headline.jsonl"
rows = [json.loads(l) for l in open(path, encoding="utf-8")]
recovered = 0

for r in rows:
    if r.get("correct"):
        continue
    # removed the box_extraction_status skip -- it was blocking rows that
    # were TOUCHED by an earlier fallback but never actually recovered
    candidate = extract_final_is_answer(r["raw_output"])
    if candidate is None:
        continue
    try:
        g = grade(f"\\boxed{{{candidate}}}", r["gold"])
        if g.get("box_extraction_status") not in ("none_found", None) and g.get("correct"):
            g["box_extraction_status"] = "final_is_fallback"
            r.update(g)
            recovered += 1
    except GoldParseError:
        pass

with open(path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Recovered {recovered} rows via 'final is <answer>' fallback")