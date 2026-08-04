# scripts/diagnose_wrong_answers.py
import json
import re

rows = [json.loads(l) for l in open("results/gen_number_theory_headline.jsonl")]

K = 15  # words to check from the tail

def tail_words(text, k):
    return " ".join(text.split()[-k:])

buckets = {}
genuinely_absent = []

for r in rows:
    if r.get("correct"):
        continue
    key = (r["condition"], r["model"])
    buckets.setdefault(key, {"gold_in_tail": 0, "absent": 0})

    gold = str(r["gold"]).strip()
    tail = tail_words(r["raw_output"], K)

    pattern = r"(?<![\w.])" + re.escape(gold) + r"(?![\w.])"
    if gold and re.search(pattern, tail):
        buckets[key]["gold_in_tail"] += 1
    else:
        buckets[key]["absent"] += 1
        genuinely_absent.append(r)

order = ["C", "A", "B1", "B2", "B3", "B4"]
for model in ["gemma4-e4b", "qwen3-1.7b"]:
    print(f"--- {model} ---")
    for c in order:
        b = buckets.get((c, model), {})
        print(f"{c}: gold_in_tail(unparsed-but-right)={b.get('gold_in_tail', 0)}  absent(genuinely_wrong)={b.get('absent', 0)}")

print()
print(f"TOTAL genuinely absent (wrong, gold not in last {K} words): {len(genuinely_absent)}")