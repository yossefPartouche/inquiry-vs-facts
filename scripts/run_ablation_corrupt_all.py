import json

from src.models import load_model, generate
from src.schema import make_row, append_jsonl

SEED = 0
MODEL_KEY = "gemma4-e4b"
MAX_TOKENS = 1024
CONTENT_PATH = "data/ablation_content/number_theory_A_corrupt_all.jsonl"
OUTPUT_PATH = "results/gen_number_theory_ablation_all.jsonl"

def _count_tokens(model_key, handle, prompt):
    if model_key == "gemma4-e4b":
        return len(handle["tokenizer"].encode(prompt))
    return len(handle["tok"](prompt)["input_ids"])

def build_prompt_from_corrupted(corrupted_sub_steps):
    lines = []
    for s in corrupted_sub_steps:
        lines.append(f"Q{s['step']}: {s['sub_question']}")
        lines.append(f"A{s['step']}: {s['sub_answer']}")
    lines.append("Final answer: \\boxed{")
    return "\n".join(lines)

def main(limit=None):
    headline = [json.loads(l) for l in open("results/gen_number_theory_headline.jsonl")]
    content = [json.loads(l) for l in open(CONTENT_PATH)]
    content_by_id = {r["problem_id"]: r for r in content}

    pool = [r for r in headline if r["condition"] == "A" and r["model"] == MODEL_KEY and r.get("correct")]
    if limit:
        pool = pool[:limit]

    print(f"loading {MODEL_KEY}...")
    handle = load_model(MODEL_KEY)

    written = 0
    for r in pool:
        entry = content_by_id.get(r["problem_id"])
        if entry is None:
            print(f"  skip {r['problem_id']}: no corrupted content")
            continue

        prompt = build_prompt_from_corrupted(entry["corrupted_sub_steps"])
        raw_output = generate(MODEL_KEY, handle, prompt, MAX_TOKENS)
        prompt_tokens = _count_tokens(MODEL_KEY, handle, prompt)

        row = make_row(
            problem_id=r["problem_id"],
            subject=r["subject"],
            level=r["level"],
            condition="A_corrupt_all",
            model=MODEL_KEY,
            seed=SEED,
            prompt_tokens=prompt_tokens,
            raw_output=raw_output,
            gold=r["gold"],
            finish_reason="ok",
        )
        append_jsonl(row, OUTPUT_PATH)
        written += 1
        print(f"  done {r['problem_id']} ({written}/{len(pool)})")

    print(f"\nWrote {written} rows -> {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
