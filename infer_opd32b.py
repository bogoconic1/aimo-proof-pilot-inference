#!/usr/bin/env python
"""Simple inference for the opd-32b-deploy model (Olmo3SinkForCausalLM).

The checkpoint is a standard Olmo3 32B plus one extra trained parameter per
layer: `self_attn.sinks` — a per-head attention-sink logit (gpt-oss style).
Vanilla transformers Olmo3 doesn't know about it, so this script:
  1. adds a `sinks` parameter to Olmo3Attention, and
  2. registers an eager attention function that appends the sink logit as an
     extra softmax column (absorbing probability mass) before dropping it.
Everything else (hybrid SWA, YaRN rope, chat template) is stock Olmo3.

Usage:
  python infer_opd32b.py [--max-new-tokens 4096] [--problem "..."]
"""
import argparse
import time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import transformers.models.olmo3.modeling_olmo3 as olmo3
from transformers.models.olmo3.modeling_olmo3 import repeat_kv

MODEL_DIR = "/workspace/models/opd-32b-deploy"

SAMPLE_PROBLEMS = [
    "Prove that the square root of 2 is irrational.",
    "Let $a$, $b$, $c$ be positive reals with $abc = 1$. Prove that "
    "$a^2 + b^2 + c^2 \\ge a + b + c$.",
    "Find all pairs of positive integers $(x, y)$ such that $x^2 - y^2 = 45$, "
    "and prove your list is complete.",
]


def eager_sink_attention(module, query, key, value, attention_mask, scaling,
                         dropout=0.0, **kwargs):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # gpt-oss style sink: one extra logit per head that soaks up probability
    # mass and is dropped after the softmax.
    bsz, _, q_len, _ = attn_weights.shape
    sinks = module.sinks.reshape(1, -1, 1, 1).expand(bsz, -1, q_len, 1)
    combined = torch.cat([attn_weights, sinks.to(attn_weights.dtype)], dim=-1)
    probs = nn.functional.softmax(combined, dim=-1, dtype=torch.float32)
    attn_weights = probs[..., :-1].to(query.dtype)

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def install_sink_support():
    ALL_ATTENTION_FUNCTIONS.register("eager_sink", eager_sink_attention)
    orig_init = olmo3.Olmo3Attention.__init__

    def init_with_sinks(self, config, layer_idx):
        orig_init(self, config, layer_idx)
        self.sinks = nn.Parameter(torch.empty(config.num_attention_heads))

    olmo3.Olmo3Attention.__init__ = init_with_sinks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--problem", action="append", default=None,
                    help="problem text; repeatable (default: built-in samples)")
    ap.add_argument("--no-stream", action="store_true")
    args = ap.parse_args()

    install_sink_support()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"loading {args.model} (bf16, ~61 GB) ...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager_sink",
    )
    model.eval()
    print(f"loaded in {time.time() - t0:.0f}s | sinks[0][:4] = "
          f"{model.model.layers[0].self_attn.sinks[:4].tolist()}", flush=True)

    problems = args.problem or SAMPLE_PROBLEMS
    streamer = None if args.no_stream else TextStreamer(tokenizer, skip_prompt=True)

    for i, problem in enumerate(problems, 1):
        print(f"\n{'=' * 80}\nPROBLEM {i}: {problem}\n{'=' * 80}", flush=True)
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                streamer=streamer,
            )
        gen = out[0, inputs["input_ids"].shape[1]:]
        dt = time.time() - t0
        text = tokenizer.decode(gen, skip_special_tokens=True)
        think, sep, answer = text.partition("</think>")
        if args.no_stream:
            print(text, flush=True)
        print(f"\n--- {len(gen)} tokens in {dt:.0f}s ({len(gen) / dt:.1f} tok/s) | "
              f"finished: {bool(sep) and len(gen) < args.max_new_tokens} ---", flush=True)


if __name__ == "__main__":
    main()
