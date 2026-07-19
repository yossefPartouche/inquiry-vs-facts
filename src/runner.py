"""
Wires the pipeline together: problem set -> build_prompt() -> generate()
-> make_row() -> append_jsonl(). Every row goes through make_row(); nothing
is ever hand-written into the results file (the discipline that saved us
after the baseline-migration mess).

B3/B4 are skipped (not fatal) until real per-problem content exists --
build_prompt() raises MissingContentError for them, which we catch here.
"""
import json

from src.data_loader import load_number_theory_problems
from src.conditions import build_prompt, MissingContentError
from src.models import load_model, generate, MODELS
from src.schema import make_row, append_jsonl, gen_path

SEED = 0  # greedy decoding is deterministic; seed is logged for the record


def _count_tokens(model_key, handle, prompt):
    """Backend-aware token count for the prompt_tokens field."""
    if model_key == "gemma4-e4b":
        return len(handle["tokenizer"].encode(prompt))
    return len(handle["tok"](prompt)["input_ids"])


def load_problem_set(path="data/problem_sets/number_theory_L1-2.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run(
    conditions=("C", "B1"),          # pilot default, per the plan's next step
    model_keys=tuple(MODELS.keys()),
    problems=None,
    max_tokens=512,
    output_path=None,
):
    problems = problems if problems is not None else load_problem_set()
    output_path = output_path or gen_path("number_theory")

    for model_key in model_keys:
        print(f"loading {model_key}...")
        handle = load_model(model_key)

        for problem in problems:
            for condition in conditions:
                try:
                    prompt = build_prompt(problem, condition)
                except MissingContentError as e:
                    print(f"  skip {problem['problem_id']} / {condition}: {e}")
                    continue

                raw_output = generate(model_key, handle, prompt, max_tokens)
                prompt_tokens = _count_tokens(model_key, handle, prompt)

                row = make_row(
                    problem_id=problem["problem_id"],
                    subject=problem["subject"],
                    level=problem["level"],
                    condition=condition,
                    model=model_key,
                    seed=SEED,
                    prompt_tokens=prompt_tokens,
                    raw_output=raw_output,
                    gold=problem["gold"],
                    finish_reason="ok",  # TODO: detect truncation properly later
                )
                append_jsonl(row, output_path)

        print(f"done with {model_key}")


if __name__ == "__main__":
    problems = load_problem_set()
    run(conditions=("C", "B1"), problems=problems, max_tokens=1024,
        output_path="results/gen_number_theory_pilot.jsonl")
    print("pilot run complete -> results/gen_number_theory_pilot.jsonl")