# run: PYTHONPATH=. python scripts/build_swap_pairs.py
import json
import random

headline = [json.loads(l) for l in open("results/gen_number_theory_headline.jsonl")]
pool = [r for r in headline if r["condition"] == "A" and r["model"] == "gemma4-e4b" and r.get("correct")]

rng = random.Random(0)
pool_ids = [r["problem_id"] for r in pool]

pairs = {}
shuffled = pool_ids[:]
rng.shuffle(shuffled)

# derangement-style pairing: shift by one so nothing pairs with itself
for i, pid in enumerate(pool_ids):
    partner = shuffled[i]
    if partner == pid:
        partner = shuffled[(i + 1) % len(shuffled)]
    pairs[pid] = partner

with open("data/ablation_content/number_theory_A_swap_pairs.jsonl", "w") as f:
    for pid, partner in pairs.items():
        f.write(json.dumps({"problem_id": pid, "swap_partner_id": partner}) + "\n")

print(f"{len(pairs)} pairs written")