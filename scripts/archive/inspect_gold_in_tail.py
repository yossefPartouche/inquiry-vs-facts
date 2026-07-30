import json, re

rows = [json.loads(l) for l in open("results/gen_number_theory_headline.jsonl")]

def tail_words(text, k=15):
    return " ".join(text.split()[-k:])

for r in rows:
    if r["condition"] != "C" or r["model"] != "gemma4-e4b" or r.get("correct"):
        continue
    gold = str(r["gold"]).strip()
    tail = tail_words(r["raw_output"])
    pattern = r"(?<![\w.])" + re.escape(gold) + r"(?![\w.])"
    if gold and re.search(pattern, tail):
        print(f"--- {r['problem_id']} | gold={gold} ---")
        print(r["raw_output"][-250:])
        print()