import json

from src.models import load_model, generate
from src.schema import make_row, append_jsonl
from scripts.ablation_corrupt import build_corrupt_last_prompt

SEED = 0
MODEL_KEY = "gemma4-e4b"
MAX_TOKENS = 1024
OUTPUT_PATH = "results/gen_number_theory_ablation.jsonl"

def _count_tokens(model_key, handle, prompt):
    if model_key == "gemma4-e4b":
        return len(handle["tokenizer"].encode(prompt))
    return len(handle["tok"](prompt)["input_ids"])

def main():
    rows = [json.loads(l) for l in open("results/gen_number_theory_headline.jsonl")]
    pool = [r for r in rows if r["condition"] == "A" and r["model"] == MODEL_KEY and r.get("correct")]

    print(f"loading {MODEL_KEY}...")
    handle = load_model(MODEL_KEY)

    skipped = 0
    written = 0

    for r in pool:
        doctored_prompt = build_corrupt_last_prompt(r["sub_steps"], seed=SEED)
        if doctored_prompt is None:
            skipped += 1
            print(f"  skip {r['problem_id']}: could not corrupt (no number found in last step)")
            continue

        raw_output = generate(MODEL_KEY, handle, doctored_prompt, MAX_TOKENS)
        prompt_tokens = _count_tokens(MODEL_KEY, handle, doctored_prompt)

        row = make_row(
            problem_id=r["problem_id"],
            subject=r["subject"],
            level=r["level"],
            condition="A_corrupt",     # per schema.py's ABLATION_CONDITIONS
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

    print(f"\nWrote {written} rows, skipped {skipped} -> {OUTPUT_PATH}")

if __name__ == "__main__":
    main()