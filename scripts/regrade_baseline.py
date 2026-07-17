# scripts/regrade_baseline.py
import json, re, sys
from src.grader import grade, GoldParseError

# Baseline-only extraction adapter: base models emit "The answer is N",
# not \boxed{N}, because the Wei et al. few-shot format dominates pretraining.
# This rewrites it so the grader can read it. Experiment models are
# instruction-tuned and emit \boxed{} directly, so this never touches real rows.
def adapt(text):
    if "\\boxed{" in text:
        return text
    m = re.findall(r"[Tt]he answer is\s*\$?(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        n = m[-1].replace(",", "")   # last one, strip thousands separators
        return text + f"\n\\boxed{{{n}}}"
    return text

path = sys.argv[1]
rows = [json.loads(l) for l in open(path) if l.strip()]
c = n = 0
from collections import Counter
statuses = Counter()
for r in rows:
    try:
        g = grade(adapt(r["raw_output"]), r["gold"])
    except GoldParseError:
        continue
    r.update(g)
    statuses[g.get("box_extraction_status")] += 1
    if g.get("correct") is not None:
        n += 1
        c += bool(g["correct"])
with open(path, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"accuracy: {c/n:.4f}  ({c}/{n})")
print("status:", dict(statuses))