"""Generate proof candidates for IMO-ProofBench problems.

Reads a proofbench CSV, prompts a model (via OpenAI-compatible endpoint) with k
candidates per problem, concurrently, and writes runs/<run_id>/responses.jsonl
(one row per problem, candidates grouped). For crash safety it also appends each
finished candidate to candidates_raw.jsonl and resumes from it on re-run.

Example (local SGLang Olmo-3-7B-Think, full bench, k=8, 7-way concurrency):
  python run_eval.py \
    --data ../data/proofbench_v2.csv \
    --base-url http://127.0.0.1:30000/v1 --served-model default \
    --model-name olmo3-7b-think --condition notool \
    --k 8 --temperature 0.7 --max-tokens 32768 --concurrency 7
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import ChatClient, REASONING  # noqa: E402
from tool_loop import run_proof_with_tools, _py_tool  # noqa: E402

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent


def repro_info() -> dict:
    """Pin the code version: git commit + dirty flag + sha256 of the harness files
    whose behaviour the run depends on (so an uncommitted run is still identifiable)."""
    def _git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=str(HERE),
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:  # noqa: BLE001
            return None
    files = ["run_eval.py", "client.py", "tool_loop.py"]
    shas = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()[:16]
            for f in files if (HERE / f).exists()}
    return {"git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "harness_sha256": shas}


def subset_of(pid: str) -> str:
    return "advanced" if "Advanced" in pid else "basic"


def load_prompt(condition: str) -> str:
    return (EVAL_ROOT / "prompts" / f"prover_{condition}.md").read_text()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--served-model", default="default")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--condition", default="notool", choices=["notool", "pytool"])
    ap.add_argument("--reasoning", default="default", choices=list(REASONING),
                    help="DeepSeek reasoning control: default|no_think|high|max")
    ap.add_argument("--max-turns", type=int, default=32, help="pytool: hard turn backstop")
    ap.add_argument("--max-tool-calls", type=int, default=24,
                    help="pytool: per-problem tool-call budget (shown to the model)")
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--concurrency", type=int, default=1, help="in-flight requests")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        sys.exit(f"--api-key-env {args.api_key_env} is set but env var is empty")

    client = ChatClient(args.base_url, args.served_model, api_key=api_key, timeout=3600.0)
    if not args.api_key_env and not client.health():
        print("[warn] /health not OK — is the SGLang server up? continuing anyway")

    prompt_tpl = load_prompt(args.condition)
    df = pd.read_csv(args.data)
    if args.limit:
        df = df.head(args.limit)
    order = list(df["Problem ID"])

    run_id = args.run_id or f"{args.model_name}__{args.condition}"
    out_dir = EVAL_ROOT / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "candidates_raw.jsonl"
    resp_path = out_dir / "responses.jsonl"

    (out_dir / "run_meta.json").write_text(json.dumps({
        "model_name": args.model_name, "served_model": args.served_model,
        "base_url": args.base_url, "condition": args.condition, "data": str(args.data),
        "k": args.k, "temperature": args.temperature, "top_p": args.top_p,
        "max_tokens": args.max_tokens, "seed": args.seed, "concurrency": args.concurrency,
        "reasoning": args.reasoning, "max_turns": args.max_turns,
        "max_tool_calls": args.max_tool_calls,
        "prompt_file": f"prover_{args.condition}.md",
        # exact tool schema sent (incl the budget in the description); rendered prompt
        # lives in each candidate's messages[0], so the template itself isn't duplicated
        "tools": _py_tool(args.max_tool_calls) if args.condition == "pytool" else None,
        "repro": repro_info(),
    }, indent=2, ensure_ascii=False))

    # resume: which (pid, j) are already done?
    done: dict[tuple[str, int], dict] = {}
    if raw_path.exists():
        for line in raw_path.open():
            r = json.loads(line)
            done[(r["problem_id"], r["j"])] = r
        print(f"[resume] {len(done)} candidates already present in {raw_path.name}")

    rows = {r["Problem ID"]: r for _, r in df.iterrows()}
    tasks = [(pid, j) for pid in order for j in range(args.k) if (pid, j) not in done]
    print(f"[run] {run_id}: {len(order)} problems x k={args.k} = {len(order)*args.k} "
          f"candidates, {len(tasks)} to do, concurrency={args.concurrency}")

    lock = threading.Lock()
    raw_f = raw_path.open("a")
    counter = {"done": len(done), "total": len(order) * args.k}

    def work(task):
        pid, j = task
        row = rows[pid]
        user_content = prompt_tpl.format(problem=row["Problem"])
        if args.condition == "pytool":
            out = run_proof_with_tools(client, user_content, reasoning=args.reasoning,
                                       max_tokens=args.max_tokens, max_turns=args.max_turns,
                                       max_tool_calls=args.max_tool_calls)
        else:
            out = client.chat([{"role": "user", "content": user_content}],
                              temperature=args.temperature, top_p=args.top_p,
                              max_tokens=args.max_tokens, seed=args.seed + j,
                              reasoning=args.reasoning)
            # lossless: archive the full exchange incl the model's thinking (reasoning_content)
            out["messages"] = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": out["text"],
                 "reasoning_content": out.get("reasoning_content") or "",
                 "finish_reason": out["finish_reason"]},
            ]
        rec = {"problem_id": pid, "j": j, "subset": subset_of(pid),
               "category": row["Category"], "level": row["Level"],
               "problem": row["Problem"], **out}
        with lock:
            raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            raw_f.flush()
            counter["done"] += 1
            tool_info = f", {out['n_tool_calls']} tool-calls/{out['n_turns']} turns" \
                if args.condition == "pytool" else ""
            print(f"  [{counter['done']}/{counter['total']}] {pid} c{j}: "
                  f"{out['completion_tokens']} tok, {out['latency_s']}s, "
                  f"{out['finish_reason']}{tool_info}")
        return rec

    if tasks:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(work, t) for t in tasks]
            for f in as_completed(futs):
                done[(f.result()["problem_id"], f.result()["j"])] = f.result()
    raw_f.close()

    # regroup into responses.jsonl in CSV order
    with resp_path.open("w") as f:
        for pid in order:
            cands = [done[(pid, j)] for j in range(args.k) if (pid, j) in done]
            if not cands:
                continue
            base = cands[0]
            f.write(json.dumps({
                "problem_id": pid, "subset": base["subset"],
                "category": base["category"], "level": base["level"],
                "problem": base["problem"],
                "candidates": [{"text": c["text"], "finish_reason": c["finish_reason"],
                                "completion_tokens": c["completion_tokens"],
                                "prompt_tokens": c["prompt_tokens"],
                                "latency_s": c["latency_s"],
                                "reasoning_tokens": c.get("reasoning_tokens"),
                                "n_turns": c.get("n_turns"),
                                "n_tool_calls": c.get("n_tool_calls")} for c in cands],
            }, ensure_ascii=False) + "\n")
    print(f"[done] {counter['done']}/{counter['total']} candidates -> {resp_path}")


if __name__ == "__main__":
    main()
