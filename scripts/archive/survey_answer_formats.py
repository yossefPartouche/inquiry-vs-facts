# scripts/survey_answer_formats.py
import json
import re
from collections import Counter

rows = [json.loads(l) for l in open("results/gen_number_theory_headline.jsonl")]

def tail_words(text, k=15):
    return " ".join(text.split()[-k:])

# common answer-announcing phrases to search for, case-insensitive
PHRASE_PATTERNS = [
    r"the answer is",
    r"final answer(?:\s+is)?[:\s]",
    r"answer[:\s]",
    r"therefore[,\s]",
    r"thus[,\s]",
    r"so[,\s].{0,15}=",
    r"must be",
    r"we (?:get|have|find)",
    r"equals",
    r"= *\d",  # bare "= <number>" near the end
]

candidates = []
phrase_hits = Counter()

for r in rows:
    if r.get("correct"):
        continue
    gold = str(r["gold"]).strip()
    if not gold:
        continue
    tail = tail_words(r["raw_output"])
    pattern = r"(?<![\w.])" + re.escape(gold) + r"(?![\w.])"
    if not re.search(pattern, tail):
        continue  # not a gold-in-tail case

    candidates.append(r)
    tail_lower = tail.lower()
    matched_any = False
    for phrase in PHRASE_PATTERNS:
        if re.search(phrase, tail_lower):
            phrase_hits[phrase] += 1
            matched_any = True
    if not matched_any:
        phrase_hits["<no known phrase matched>"] += 1

print(f"{len(candidates)} gold-in-tail candidates total\n")
print("Phrase frequency (a row can match multiple phrases):")
for phrase, count in phrase_hits.most_common():
    print(f"  {count:4d}  {phrase}")