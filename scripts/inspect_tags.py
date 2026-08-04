import json, random
from collections import Counter

rows = [json.loads(l) for l in open("data/problem_sets/number_theory_L1-2.jsonl")]
print(f"Total: {len(rows)}")
counts = Counter(r["problem_type"] for r in rows)
print("Distribution:", dict(counts))

random.seed(0)
for tag in counts:
    sample = random.sample([r for r in rows if r["problem_type"] == tag], min(4, counts[tag]))
    print(f"\n--- {tag} ({counts[tag]}) ---")
    for r in sample:
        print(f"  [{r['problem_id']}] {r['problem'][:150]}")