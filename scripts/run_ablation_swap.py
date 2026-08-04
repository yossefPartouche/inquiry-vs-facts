# run: PYTHONPATH=. python -c "from scripts.run_ablation_swap import main; main(limit=3)"
import json

from src.models import load_model, generate
from src.schema import make_row, append_jsonl

SEED = 0
MODEL_KEY = "gemma4-e4b"
MAX_TOKENS = 1024
PAIRS_PATH = "data/ablation_content/number_theory_A_swap_pairs.jsonl"
OUTPUT_PATH = "results/gen_number_theory_ablation_swap.jsonl"
PROBLEM_SET_PATH = "data/problem_sets/number_theory_L1-2_filtered.jsonl"

def _count_tokens(model_key, handle, prompt):
    if model_key == "gemma4-e4b":
        return len(handle["tokenizer"].encode(prompt))
    return len(handle["tok"](prompt)["input_ids"])

def build_swap_prompt(partner_sub_steps, target_problem_text):
    lines = []
    for s in partner_sub_steps:
        lines.append(f"Q{s['step']}: {s['sub_question']}")
        lines.append(f"A{s['step']}: {s['sub_answer']}")
    lines.append("")
    lines.append("Solve the following math problem. Give your final answer in the form")
    lines.append("\\boxed{<answer>}.")
    lines.append("")
    lines.append(f"Problem: {target_problem_text}")
    return "\n".join(lines)

def main(limit=None):
    headline = [json.loads(l) for l in open("results/gen_number_theory_headline.jsonl")]
    by_id = {r["problem_id"]: r for r in headline if r["condition"] == "A" and r["model"] == MODEL_KEY}

    problems = [json.loads(l) for l in open(PROBLEM_SET_PATH)]
    problem_text_by_id = {p["problem_id"]: p["problem"] for p in problems}

    pairs = [json.loads(l) for l in open(PAIRS_PATH)]
    if limit:
        pairs = pairs[:limit]

    print(f"loading {MODEL_KEY}...")
    handle = load_model(MODEL_KEY)

    written = 0
    for p in pairs:
        target = by_id[p["problem_id"]]
        partner = by_id[p["swap_partner_id"]]
        target_problem_text = problem_text_by_id[p["problem_id"]]

        prompt = build_swap_prompt(partner["sub_steps"], target_problem_text)
        raw_output = generate(MODEL_KEY, handle, prompt, MAX_TOKENS)
        prompt_tokens = _count_tokens(MODEL_KEY, handle, prompt)

        row = make_row(
            problem_id=target["problem_id"],
            subject=target["subject"],
            level=target["level"],
            condition="A_swap",
            model=MODEL_KEY,
            seed=SEED,
            prompt_tokens=prompt_tokens,
            raw_output=raw_output,
            gold=target["gold"],
            finish_reason="ok",
        )
        append_jsonl(row, OUTPUT_PATH)
        written += 1
        print(f"  done {target['problem_id']} <- swapped with {partner['problem_id']} ({written}/{len(pairs)})")

    print(f"\nWrote {written} rows -> {OUTPUT_PATH}")

if __name__ == "__main__":
    main()