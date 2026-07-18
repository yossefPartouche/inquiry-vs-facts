# src/models.py
#
# Unified model loading/generation interface for both experiment models.
# Qwen3-1.7B runs on the `transformers` backend (full precision, fits fine).
# Gemma 4 E4B runs on the `mlx_lm` backend (4-bit quantized) -- full precision
# does not fit in 16GB RAM on Apple Silicon; bitsandbytes 4-bit is CUDA-only
# and does not work here, so MLX (Apple's native framework) is the fix.
#
# Gemma 4 thinking mode is now OFF, per the plan, via enable_thinking=False
# passed to apply_chat_template(). Confirmed empirically: output no longer
# contains <|channel|>thought...</channel|> blocks.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mlx_lm import load as mlx_load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

GEMMA_MLX_NAME = "mlx-community/gemma-4-e4b-it-4bit"
QWEN_NAME = "Qwen/Qwen3-1.7B"

MODELS = {
    "gemma4-e4b": GEMMA_MLX_NAME,
    "qwen3-1.7b": QWEN_NAME,
}
# --- Qwen / transformers backend --------------------------------------------

def _load_qwen(name):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        # no device_map="auto" -- caused meta-tensor crashes on this machine;
        # load whole model onto one real device explicitly instead.
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.generation_config.pad_token_id = tok.pad_token_id
    return {"backend": "transformers", "tok": tok, "model": model}


def _generate_qwen(handle, prompt, max_tokens=512):
    tok, model = handle["tok"], handle["model"]
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_tokens,
        do_sample=False,                 # greedy, per the plan
        repetition_penalty=1.3,
        eos_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# --- Gemma / mlx_lm backend --------------------------------------------------

def _load_gemma(name):
    model, tokenizer = mlx_load(name)
    return {"backend": "mlx", "model": model, "tokenizer": tokenizer}


def _generate_gemma(handle, prompt, max_tokens=512):
    model, tokenizer = handle["model"], handle["tokenizer"]
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    sampler = make_sampler(temp=0.0)     # greedy, per the plan
    logits_processors = make_logits_processors(repetition_penalty=1.3)
    return mlx_generate(
        model, tokenizer, prompt=formatted, max_tokens=max_tokens,
        sampler=sampler, logits_processors=logits_processors, verbose=False,
    )


# --- Unified interface -------------------------------------------------------

def load_model(model_key):
    """model_key: one of MODELS' keys, e.g. 'gemma4-e4b' or 'qwen3-1.7b'
    (matches src/schema.py's MODELS enum -- keep them in sync)."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model_key {model_key!r}. Expected one of {list(MODELS)}")
    hf_name = MODELS[model_key]
    if model_key == "gemma4-e4b":
        return _load_gemma(hf_name)
    return _load_qwen(hf_name)


def generate(model_key, handle, prompt, max_tokens=512):
    if model_key == "gemma4-e4b":
        return _generate_gemma(handle, prompt, max_tokens)
    return _generate_qwen(handle, prompt, max_tokens)


if __name__ == "__main__":
    test_prompt = "What is 2+2? Answer with \\boxed{answer}."
    for key in MODELS:
        print(f"loading {key} ({MODELS[key]})...")
        handle = load_model(key)
        print(generate(key, handle, test_prompt))
        print("-" * 40)