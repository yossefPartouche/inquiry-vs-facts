# scripts/debug_fallback_on_real_row.py
import json, re

def extract_final_is_answer(raw_output):
    matches = re.findall(
        r"\bis\s+[\*'\"]{0,2}([\-\d./\\{}^a-zA-Z\s]+?)[\*'\"]{0,2}\.\s*(?:$|\n)",
        raw_output
    )
    if not matches:
        return None
    return matches[-1].strip()

rows = [json.loads(l) for l in open("results/gen_number_theory_headline.jsonl")]

r = next(x for x in rows
         if x["problem_id"] == "number_theory__2__0029"
         and x["condition"] == "C"
         and x["model"] == "gemma4-e4b")

print("correct:", r.get("correct"))
print("box_extraction_status:", r.get("box_extraction_status"))
print("raw tail:", repr(r["raw_output"][-150:]))
print()
print("extracted:", extract_final_is_answer(r["raw_output"]))