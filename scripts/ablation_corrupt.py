# scripts/ablation_corrupt.py
import json
import re
import random

def corrupt_numeric_value(original_text, seed=None):
    """
    Given a sub_answer string, find the LAST standalone number/fraction in it
    (the actual computed result) and replace it with a plausible-but-wrong
    nearby value. Returns the corrupted text, or None if no number found.
    """
    rng = random.Random(seed)

    # find standalone numbers (ints, decimals, simple fractions like a/b)
    matches = list(re.finditer(r'(?<![\w.])(-?\d+(?:\.\d+)?(?:/\d+)?)(?![\w.])', original_text))
    if not matches:
        return None

    last = matches[-1]
    original_value = last.group(1)

    # generate a plausible wrong replacement: shift by a small random amount
    try:
        if '/' in original_value:
            num, denom = original_value.split('/')
            new_num = int(num) + rng.choice([-2, -1, 1, 2])
            replacement = f"{new_num}/{denom}"
        elif '.' in original_value:
            val = float(original_value)
            replacement = str(round(val + rng.choice([-1, -0.5, 0.5, 1]), 2))
        else:
            val = int(original_value)
            replacement = str(val + rng.choice([-3, -2, -1, 1, 2, 3]))
    except (ValueError, ZeroDivisionError):
        return None

    corrupted = original_text[:last.start(1)] + replacement + original_text[last.end(1):]
    return corrupted


def build_corrupt_last_prompt(sub_steps, seed=None):
    """
    Takes parsed sub_steps, corrupts the LAST step's sub_answer, and
    re-renders the transcript up to (but not including) the final answer --
    ready to feed back to the model to force a fresh completion.
    """
    steps = [dict(s) for s in sub_steps]  # copy, don't mutate original
    last = steps[-1]
    corrupted_answer = corrupt_numeric_value(last["sub_answer"], seed=seed)
    if corrupted_answer is None:
        return None  # couldn't find a number to corrupt in the last step

    last["sub_answer"] = corrupted_answer

    lines = []
    for s in steps:
        lines.append(f"Q{s['step']}: {s['sub_question']}")
        lines.append(f"A{s['step']}: {s['sub_answer']}")
    lines.append("Final answer: \\boxed{")

    return "\n".join(lines)


if __name__ == "__main__":
    rows = [json.loads(l) for l in open("results/gen_number_theory_headline.jsonl")]
    r = next(x for x in rows if x["condition"] == "A" and x["model"] == "gemma4-e4b"
             and x["problem_id"] == "number_theory__2__0001")

    print("=== ORIGINAL last step ===")
    print(r["sub_steps"][-1])
    print()

    doctored_prompt = build_corrupt_last_prompt(r["sub_steps"], seed=0)
    print("=== DOCTORED PROMPT (fed back to model) ===")
    print(doctored_prompt)