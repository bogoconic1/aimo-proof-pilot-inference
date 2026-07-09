#!/usr/bin/env python
"""Dual-GPU batched inference for opd-32b-deploy (Olmo3 + attention sinks).

The 61 GB bf16 model fits on a single H200, so the fastest use of 2 GPUs is
data parallelism: one full model replica per GPU, each worker generating its
share of problems as one left-padded batch. Sink support is the same
eager-attention patch as infer_opd32b.py.

Usage:
  python infer_opd32b_multi.py [--gpus 0,1] [--max-new-tokens 4096]
                               [--problem "..." --problem "..."]
"""
import argparse
import json
import os
import time

import torch
import torch.multiprocessing as mp
import torch.nn as nn

MODEL_DIR = "/workspace/models/opd-32b-deploy"

SAMPLE_PROBLEMS = [
    "Prove that the square root of 2 is irrational.",
    "Let $a$, $b$, $c$ be positive reals with $abc = 1$. Prove that "
    "$a^2 + b^2 + c^2 \\ge a + b + c$.",
    "Find all pairs of positive integers $(x, y)$ such that $x^2 - y^2 = 45$, "
    "and prove your list is complete.",
    "Prove that for every integer $n \\ge 1$, the number $n^3 - n$ is divisible by 6.",
    "Prove that in any group of 13 people, at least two were born in the same month.",
    "Show that the sum $1 + \\frac{1}{2} + \\frac{1}{3} + \\cdots + \\frac{1}{n}$ "
    "is never an integer for $n \\ge 2$.",
]


def install_sink_support():
    """Add gpt-oss-style attention sinks to transformers' Olmo3."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    import transformers.models.olmo3.modeling_olmo3 as olmo3
    from transformers.models.olmo3.modeling_olmo3 import repeat_kv

    def eager_sink_attention(module, query, key, value, attention_mask, scaling,
                             dropout=0.0, **kwargs):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        bsz, _, q_len, _ = attn_weights.shape
        sinks = module.sinks.reshape(1, -1, 1, 1).expand(bsz, -1, q_len, 1)
        combined = torch.cat([attn_weights, sinks.to(attn_weights.dtype)], dim=-1)
        probs = nn.functional.softmax(combined, dim=-1, dtype=torch.float32)
        attn_weights = probs[..., :-1].to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
        return attn_output, attn_weights

    ALL_ATTENTION_FUNCTIONS.register("eager_sink", eager_sink_attention)
    orig_init = olmo3.Olmo3Attention.__init__

    def init_with_sinks(self, config, layer_idx):
        orig_init(self, config, layer_idx)
        self.sinks = nn.Parameter(torch.empty(config.num_attention_heads))

    olmo3.Olmo3Attention.__init__ = init_with_sinks


def worker(gpu_id, model_dir, problems, gen_args, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    install_sink_support()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 2  # per config.json

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="eager_sink",
    )
    model.eval()
    result_queue.put(("status", gpu_id, f"model loaded in {time.time() - t0:.0f}s, "
                      f"generating {len(problems)} problems as one batch"))

    # chat template already emits bos, so don't add special tokens again
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for _, p in problems
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                       add_special_tokens=False).to(model.device)

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=gen_args["max_new_tokens"],
            do_sample=True,
            temperature=gen_args["temperature"],
            top_p=gen_args["top_p"],
        )
    dt = time.time() - t0

    prompt_len = inputs["input_ids"].shape[1]
    total_new = 0
    for (idx, problem), row in zip(problems, out):
        gen = row[prompt_len:]
        n_new = int((gen != tokenizer.pad_token_id).sum())
        total_new += n_new
        text = tokenizer.decode(gen, skip_special_tokens=True)
        think, sep, answer = text.partition("</think>")
        result_queue.put(("result", gpu_id, {
            "idx": idx, "problem": problem,
            "answer": answer.strip() if sep else text,
            "think_chars": len(think),
            "tokens": n_new,
            "finished": bool(sep) and n_new < gen_args["max_new_tokens"],
        }))
    result_queue.put(("done", gpu_id, {"batch_tokens": total_new, "seconds": dt,
                                       "tok_s": total_new / dt}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--problem", action="append", default=None)
    ap.add_argument("--json-out", default=None, help="also write results as JSON")
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    problems = list(enumerate(args.problem or SAMPLE_PROBLEMS))
    shards = {g: problems[i::len(gpus)] for i, g in enumerate(gpus)}
    gen_args = {"max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature, "top_p": args.top_p}

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(g, args.model, shards[g], gen_args, queue))
             for g in gpus if shards[g]]
    t0 = time.time()
    for p in procs:
        p.start()

    results, done_stats, pending = {}, {}, len(procs)
    while pending:
        kind, gpu_id, payload = queue.get()
        if kind == "status":
            print(f"[gpu{gpu_id}] {payload}", flush=True)
        elif kind == "result":
            results[payload["idx"]] = payload
            print(f"[gpu{gpu_id}] problem {payload['idx'] + 1} done: "
                  f"{payload['tokens']} tokens, finished={payload['finished']}", flush=True)
        elif kind == "done":
            done_stats[gpu_id] = payload
            pending -= 1
            print(f"[gpu{gpu_id}] batch: {payload['batch_tokens']} tokens in "
                  f"{payload['seconds']:.0f}s ({payload['tok_s']:.1f} tok/s)", flush=True)
    for p in procs:
        p.join()
    wall = time.time() - t0

    for idx in sorted(results):
        r = results[idx]
        print(f"\n{'=' * 80}\nPROBLEM {idx + 1}: {r['problem']}\n"
              f"(thought for {r['think_chars']} chars, answer below)\n{'-' * 80}\n"
              f"{r['answer']}", flush=True)

    total = sum(s["batch_tokens"] for s in done_stats.values())
    print(f"\n{'=' * 80}\nTOTAL: {len(results)} problems, {total} new tokens, "
          f"{wall:.0f}s wall (incl. load) -> {total / wall:.1f} tok/s aggregate; "
          f"per-GPU gen speed: "
          + ", ".join(f"gpu{g}={s['tok_s']:.1f} tok/s" for g, s in sorted(done_stats.items())),
          flush=True)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump([results[i] for i in sorted(results)], f, indent=2, ensure_ascii=False)
        print(f"results written to {args.json_out}")


if __name__ == "__main__":
    main()
