# scripts/test_parse_sub_steps.py
import json
from scripts.parse_sub_steps import parse_sub_steps

rows = [json.loads(l) for l in open("results/gen_number_theory_headline.jsonl")]
a_rows = [r for r in rows if r["condition"] == "A" and r["model"] == "gemma4-e4b"]

sample = a_rows[:10]  # first 10, or use random.sample for a spread

for r in sample:
    steps = parse_sub_steps(r["raw_output"])
    print(f"{r['problem_id']}: {len(steps)} steps parsed", end="")
    if not steps:
        print("  <-- ZERO STEPS, likely a parsing failure, needs inspection")
    else:
        print()
        # flag anything suspicious: empty question/answer, or a step that
        # looks truncated (very short answer, e.g. under 5 chars)
        for s in steps:
            if not s["sub_question"] or not s["sub_answer"]:
                print(f"    !! step {s['step']} has an EMPTY question or answer")
            if len(s["sub_answer"]) < 5:
                print(f"    !! step {s['step']} has a SUSPICIOUSLY SHORT answer: {s['sub_answer']!r}")