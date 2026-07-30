# scripts/inspect_no_phrase_match.py
import json, re

rows = [json.loads(l) for l in open("results/gen_number_theory_headline.jsonl")]

PHRASE_PATTERNS = [
    r"the answer is", r"final answer(?:\s+is)?[:\s]", r"answer[:\s]",
    r"therefore[,\s]", r"thus[,\s]", r"so[,\s].{0,15}=", r"must be",
    r"we (?:get|have|find)", r"equals", r"= *\d",
]

def tail_words(text, k=15):
    return " ".join(text.split()[-k:])

count = 0
for r in rows:
    if r.get("correct"):
        continue
    gold = str(r["gold"]).strip()
    if not gold:
        continue
    tail = tail_words(r["raw_output"])
    pattern = r"(?<![\w.])" + re.escape(gold) + r"(?![\w.])"
    if not re.search(pattern, tail):
        continue
    tail_lower = tail.lower()
    if any(re.search(p, tail_lower) for p in PHRASE_PATTERNS):
        continue  # matched a known phrase, skip

    print(f"--- {r['problem_id']} | {r['condition']} | {r['model']} | gold={gold} ---")
    print(tail)
    print()
    count += 1
    if count >= 20:
        break