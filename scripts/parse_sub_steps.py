import json
import re

def parse_sub_steps(raw_output):
    """
    Parses a condition-A raw_output into structured sub_steps:
    [{"step": int, "sub_question": str, "sub_answer": str}, ...]

    Splits on Qn: markers. Everything between "Qn:" and the next "Qn+1:"
    (or "Final answer:") is that step's content; within it, everything
    before the first "An:" is the question, everything after is the answer.
    """
    # cut off anything from "Final answer:" onward -- not part of the chain
    chain_text = re.split(r"Final answer:", raw_output)[0]

    # split into Qn: ... blocks
    pieces = re.split(r"\bQ(\d+):\s*", chain_text)
    # pieces[0] is anything before the first Q1: (usually empty/whitespace)
    # then alternating: number, block_text, number, block_text, ...

    steps = []
    for i in range(1, len(pieces), 2):
        step_num = int(pieces[i])
        block = pieces[i + 1]

        # within the block, split on the matching An:
        m = re.search(rf"\bA{step_num}:\s*", block)
        if not m:
            continue  # malformed step, skip rather than guess
        sub_question = block[:m.start()].strip()
        sub_answer = block[m.end():].strip()

        steps.append({
            "step": step_num,
            "sub_question": sub_question,
            "sub_answer": sub_answer,
        })

    return steps


if __name__ == "__main__":
    rows = [json.loads(l) for l in open("results/gen_number_theory_headline.jsonl")]
    r = next(x for x in rows if x["condition"] == "A" and x["model"] == "gemma4-e4b"
             and x["problem_id"] == "number_theory__2__0001")

    steps = parse_sub_steps(r["raw_output"])
    print(f"{len(steps)} steps parsed\n")
    for s in steps:
        print(f"--- Step {s['step']} ---")
        print("Q:", s["sub_question"][:80])
        print("A:", s["sub_answer"][:80])
        print()