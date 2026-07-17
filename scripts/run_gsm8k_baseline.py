"""Throwaway GSM8K baseline runner (Track M, Week 1 gate).

Purpose: validate the grader against a published number, NOT to be the real
pipeline. Track X's runner.py replaces this. Delete after the gate passes.

Target: Qwen3-1.7B-Base, 4-shot CoT, GSM8K = 75.44
        (Qwen3 Tech Report arXiv:2505.09388, Table 8; protocol in section 3.3)

Base model, not instruct: 75.44 is a base-model number, and a base model is a
pure completion engine -- no chat template between the prompt and the result.

Usage (from project root):
    pip install transformers torch accelerate datasets
    python -m scripts.run_gsm8k_baseline --limit 20     # smoke test first
    python -m scripts.run_gsm8k_baseline               # full split (1319)
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.grader import GoldParseError, grade

ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = ROOT / "prompts" / "gsm8k_baseline_fewshot.txt"
OUT_PATH = ROOT / "results" / "baseline_gsm8k.jsonl"

MODEL = "Qwen/Qwen3-1.7B-Base"
SEED = 0
MAX_NEW_TOKENS = 1024
_ANSWER_IS = re.compile(r"[Tt]he answer is\s*\$?(-?[\d,]+(?:\.\d+)?)")

if torch.backends.mps.is_available():
    DEVICE, DTYPE = "mps", torch.float16      # bf16 is flaky on MPS
elif torch.cuda.is_available():
    DEVICE, DTYPE = "cuda", torch.bfloat16
else:
    DEVICE, DTYPE = "cpu", torch.float32

_tok = None
_model = None


def get_model():
    """Load once, lazily -- so --help doesn't pull 3.4GB of weights."""
    global _tok, _model
    if _model is None:
        print(f"loading {MODEL} on {DEVICE} ({DTYPE})...")
        _tok = AutoTokenizer.from_pretrained(MODEL)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=DTYPE
        ).to(DEVICE).eval()
    return _tok, _model

def to_boxed(text: str) -> str:
    """Baseline-only adapter. Qwen3-Base has the Wei et al. prompt memorized and
    emits 'The answer is N.' regardless of what the exemplars demonstrate. Rewrite
    that closer into \\boxed{N} so the real grader sees what it expects.
    Does NOT touch src/grader.py -- the experiment's instruct models follow the
    \\boxed{} instruction directly, so this adapter is throwaway too."""
    if "\\boxed" in text:
        return text
    m = _ANSWER_IS.search(text)
    if m:
        return text + f"\n\\boxed{{{m.group(1).replace(',', '')}}}"
    return text


def call_model(prompt: str) -> tuple[str, int]:
    """Greedy completion. Returns (raw_output, prompt_tokens)."""
    tok, model = get_model()
    ids = tok(prompt, return_tensors="pt").to(DEVICE)
    n_prompt = int(ids.input_ids.shape[1])

    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,                  # greedy
            pad_token_id=tok.eos_token_id,
        )

    text = tok.decode(out[0][n_prompt:], skip_special_tokens=True)

    # A base model answers, then cheerfully invents its own next "Q:" and keeps
    # going. Cut there. extract_boxed takes the LAST box, so without this we'd
    # grade the answer to a problem the model made up. Load-bearing.
    return text.split("\nQ:")[0].strip(), n_prompt


def load_gsm8k(limit: int | None = None) -> list[dict]:
    """GSM8K test split. Gold is the number after '#### ' in the solution."""
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split="test")
    rows = []
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        rows.append({
            "problem_id": f"gsm8k_test_{i:04d}",
            "question": ex["question"],
            "gold": gold,
        })
    return rows


def validate_golds(rows: list[dict]) -> None:
    """Golds are DATA. A gold that won't parse is a bug, not a wrong answer --
    it would deflate accuracy identically across every condition and hide there."""
    bad = [r["problem_id"] for r in rows
           if _gold_fails(r["gold"])]
    if bad:
        raise SystemExit(f"{len(bad)} unparseable golds, e.g. {bad[:5]}")
    print(f"[ok] {len(rows)} golds parse cleanly")


def _gold_fails(gold: str) -> bool:
    try:
        grade(r"\boxed{0}", gold)
        return False
    except GoldParseError:
        return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    template = PROMPT_PATH.read_text()
    if "{question}" not in template:
        raise SystemExit(f"{PROMPT_PATH} has no {{question}} placeholder")

    rows = load_gsm8k(args.limit)
    validate_golds(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_correct = n_parse_ok = 0
    t0 = time.time()

    with OUT_PATH.open("w") as f:
        for i, row in enumerate(rows, 1):
            prompt = template.replace("{question}", row["question"])
            raw, ptok = call_model(prompt)
            g = grade(to_boxed(raw), row["gold"])

            n_correct += g["correct"]
            n_parse_ok += g["parse_ok"]

            f.write(json.dumps({
                "problem_id": row["problem_id"],
                "subject": "gsm8k",
                "level": None,
                "condition": "baseline_4shot_cot",
                "rung_label": None,
                "model": MODEL,
                "seed": SEED,
                "prompt_tokens": ptok,
                "raw_output": raw,                      # never overwrite:
                "parsed_answer": g["parsed_answer"],    # re-grading is free,
                "gold": row["gold"],                    # re-generating is not
                "correct": g["correct"],
                "parse_ok": g["parse_ok"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }) + "\n")
            f.flush()

            if i % 10 == 0 or i == len(rows):
                print(f"{i}/{len(rows)}  acc={n_correct/i:.3f}  "
                      f"parse_ok={n_parse_ok/i:.3f}  ({time.time()-t0:.0f}s)")

    n = len(rows)
    print("\n" + "=" * 52)
    print(f"accuracy : {n_correct/n:.4f}   ({n_correct}/{n})")
    print(f"parse_ok : {n_parse_ok/n:.4f}   ({n_parse_ok}/{n})")
    print(f"target   : 0.7544   (Qwen3 Tech Report, Table 8)")
    print("=" * 52)
    print("Read parse_ok FIRST. If it's low, the prompt is broken and the")
    print("accuracy number below it is meaningless.")
    print(f"\nrows -> {OUT_PATH}")


if __name__ == "__main__":
    main()