"""Calibrate the flash grader on IMO-GradingBench (async, high-concurrency).

Grade a stratified subset of gradingbench.csv with deepseek-v4-flash under several
configs (reasoning x tool) and compare to human grades. Each gradingbench row carries
the reference solution + guidelines + a candidate Response + human Points(0-7) and
Reward(4-cat), so we grade with the same B.5 prompt we use in production and measure
agreement.

Configs: {high, max} x {notool, pytool}. "pytool" gives the grader an execute_python
tool (numpy/sympy sandbox) to verify computations/claims while grading. Per config we
report: 4-cat accuracy vs human Reward, MAE vs Points (golden floor 3.9%), Pearson,
confusion matrix, and mean token usage.

Async engine (httpx + asyncio.Semaphore) drives ~1000 concurrent requests; sandbox code
execution runs in a thread pool (run_in_executor) with a wall-clock guard. Sampling,
metrics and the grades_all.jsonl schema are unchanged from the threaded version, so runs
resume across both.

Example (200 rows, all 4 configs, flash, 1000-way):
  python calibrate_grader.py --data ../data/gradingbench.csv --n 200 \
    --base-url https://api.deepseek.com/v1 --served-model deepseek-v4-flash \
    --api-key-env DEEPSEEK_API_KEY --concurrency 1000 --max-tokens 65536
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))
from async_client import AsyncChatClient  # noqa: E402
from grader import parse_score  # noqa: E402
from safe_session import SafePythonSession  # noqa: E402

CAT = {0: "Incorrect", 1: "Partial", 6: "Almost", 7: "Correct"}
CATS = ["Incorrect", "Partial", "Almost", "Correct"]

GRADER_PY_TOOL = [{
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": ("Execute Python in a sandbox (numpy/sympy/scipy; no file/network) "
                        "to verify computations, check small cases, or test claims in the "
                        "proposed solution before you assign a score. Returns stdout."),
        "parameters": {"type": "object",
                       "properties": {"code": {"type": "string",
                                               "description": "Python code; use print()."}},
                       "required": ["code"]},
    },
}]
TOOL_NOTE = ("You have access to an `execute_python` tool (numpy/sympy/scipy sandbox). "
             "Use it to verify any computation, check small cases, or test claims in the "
             "proposed solution before deciding the score. Tool output is for your own "
             "verification only; base the final score on mathematical correctness.\n\n")

CONFIGS = [
    {"name": "high_notool", "reasoning": "high", "tool": False},
    {"name": "high_pytool", "reasoning": "high", "tool": True},
    {"name": "max_notool", "reasoning": "max", "tool": False},
    {"name": "max_pytool", "reasoning": "max", "tool": True},
]


def stratified(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Balanced sample: n/4 per human Reward category (capped by availability)."""
    per = n // len(CATS)
    parts = [df[df["Reward"] == c].sample(min(per, (df["Reward"] == c).sum()),
                                          random_state=seed) for c in CATS]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return round(num / (dx * dy), 4) if dx and dy else None


async def _exec(executor, sess, code, timeout=45.0):
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(loop.run_in_executor(executor, sess.execute, code),
                                      timeout=timeout)
    except asyncio.TimeoutError:
        return "[execution timed out]"
    except Exception as e:  # noqa: BLE001
        return f"[execution error: {e}]"


async def grade_with_tools(client, executor, prompt, reasoning, max_tokens, max_turns):
    sess = SafePythonSession(timeout=20, mem_mb=4096)
    messages = [{"role": "user", "content": prompt}]
    ctoks = ptoks = rtoks = n_tool = 0
    finish, text = "max_turns", ""
    t0 = time.monotonic()
    try:
        for _turn in range(max_turns):
            out = await client.chat_raw(messages, reasoning=reasoning, max_tokens=max_tokens,
                                        tools=GRADER_PY_TOOL)
            ptoks += out["prompt_tokens"] or 0
            ctoks += out["completion_tokens"] or 0
            rtoks += out["reasoning_tokens"] or 0
            m = out["message"]
            finish = out["finish_reason"]
            tcs = m.get("tool_calls") or []
            if finish == "tool_calls" and tcs:
                messages.append({"role": "assistant", "content": m.get("content") or "",
                                 "tool_calls": tcs})
                for c in tcs:
                    fn = c.get("function", {})
                    try:
                        code = json.loads(fn.get("arguments") or "{}").get("code", "")
                    except Exception as e:  # noqa: BLE001 - malformed tool args
                        res = f"[bad tool args: {e}]"
                    else:
                        res = await _exec(executor, sess, code)
                        n_tool += 1
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": res[:4000]})
                continue
            text = m.get("content") or ""
            break
    finally:
        sess.close()
    return {"text": text, "finish_reason": finish, "completion_tokens": ctoks,
            "prompt_tokens": ptoks, "reasoning_tokens": rtoks, "n_tool_calls": n_tool,
            "latency_s": round(time.monotonic() - t0, 2)}


async def grade_one(client, executor, tpl, row, cfg, max_tokens, max_turns):
    body = (TOOL_NOTE + tpl) if cfg["tool"] else tpl
    prompt = body.format(problem_statement=row["Problem"], solution=row["Solution"],
                         guidelines=row["Grading guidelines"], student_answer=row["Response"])
    if cfg["tool"]:
        out = await grade_with_tools(client, executor, prompt, cfg["reasoning"],
                                     max_tokens, max_turns)
    else:
        out = await client.chat([{"role": "user", "content": prompt}],
                                reasoning=cfg["reasoning"], max_tokens=max_tokens)
    g = parse_score(out["text"])
    return {
        "grading_id": row["Grading ID"], "problem_id": row["Problem ID"], "config": cfg["name"],
        "points": int(row["Points"]), "reward": row["Reward"],
        "score": g["score"], "cat": CAT.get(g["score"]),
        "finish_reason": out["finish_reason"],
        "completion_tokens": out.get("completion_tokens"),
        "reasoning_tokens": out.get("reasoning_tokens"),
        "n_tool_calls": out.get("n_tool_calls", 0),
        "latency_s": out.get("latency_s"),
    }


def summarize(recs: list[dict]) -> dict:
    ok = [r for r in recs if r["score"] is not None]
    n, cov = len(recs), len(ok)
    if not ok:
        return {"n": n, "parsed": 0}
    conf = {h: {g: 0 for g in CATS} for h in CATS}
    for r in ok:
        conf[r["reward"]][r["cat"]] += 1
    mae = sum(abs(r["score"] - r["points"]) for r in ok) / cov
    return {
        "n": n, "parsed": cov, "coverage": round(cov / n, 3),
        "accuracy_4cat": round(sum(r["cat"] == r["reward"] for r in ok) / cov, 4),
        "mae": round(mae, 3), "mae_pct": round(mae / 7 * 100, 2),
        "pearson": pearson([r["score"] for r in ok], [r["points"] for r in ok]),
        "mean_completion_tokens": round(sum(r["completion_tokens"] or 0 for r in ok) / cov),
        "mean_reasoning_tokens": round(sum(r["reasoning_tokens"] or 0 for r in ok) / cov),
        "mean_tool_calls": round(sum(r["n_tool_calls"] or 0 for r in ok) / cov, 2),
        "confusion_human_x_grader": conf,
    }


async def amain(args) -> None:
    key = os.environ.get(args.api_key_env)
    if not key:
        sys.exit(f"empty {args.api_key_env}")
    configs = [c for c in CONFIGS if c["name"] in args.configs.split(",")]
    tpl = (EVAL_ROOT / "prompts" / "grader.md").read_text()

    df = pd.read_csv(args.data)
    sub = stratified(df, args.n, args.seed)
    if args.limit:
        sub = sub.head(args.limit)
    rows = {r["Grading ID"]: r for _, r in sub.iterrows()}

    out_dir = EVAL_ROOT / "runs" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "grades_all.jsonl"
    done = set()
    if raw_path.exists():
        for line in raw_path.open():
            r = json.loads(line)
            done.add((r["grading_id"], r["config"]))
        print(f"[resume] {len(done)} (row,config) already graded")
    tasks = [(gid, c) for gid in rows for c in configs if (gid, c["name"]) not in done]
    print(f"[calib] {len(sub)} rows x {len(configs)} configs = {len(sub)*len(configs)} "
          f"({len(tasks)} to do), concurrency={args.concurrency}")
    print("  human Reward in sample:", sub["Reward"].value_counts().to_dict())

    client = AsyncChatClient(args.base_url, args.served_model, key,
                             max_connections=args.concurrency, timeout=3600.0)
    executor = ThreadPoolExecutor(max_workers=min(args.concurrency, 256))
    sema = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    raw_f = raw_path.open("a")
    counter = {"done": 0, "total": len(tasks)}

    async def work(gid, cfg):
        async with sema:
            try:
                rec = await grade_one(client, executor, tpl, rows[gid], cfg,
                                      args.max_tokens, args.max_turns)
            except Exception as e:  # noqa: BLE001 - keep the batch alive on a bad task
                row = rows[gid]
                rec = {"grading_id": gid, "problem_id": row["Problem ID"], "config": cfg["name"],
                       "points": int(row["Points"]), "reward": row["Reward"], "score": None,
                       "cat": None, "finish_reason": "error", "error": str(e)[:200],
                       "completion_tokens": None, "reasoning_tokens": None, "n_tool_calls": 0}
        async with lock:
            raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            raw_f.flush()
            counter["done"] += 1
            print(f"  [{counter['done']}/{counter['total']}] {gid} {cfg['name']}: "
                  f"score={rec['score']} (human {rec.get('points')}/{rec.get('reward')}) "
                  f"{rec.get('completion_tokens')}tok tools={rec.get('n_tool_calls')}")
        return rec

    if tasks:
        await asyncio.gather(*[work(g, c) for g, c in tasks])
    raw_f.close()
    await client.aclose()
    executor.shutdown(wait=False)

    allrecs = [json.loads(l) for l in raw_path.open()]
    summary = {"run_id": args.run_id, "model": args.served_model, "n_rows": len(rows),
               "by_config": {c["name"]: summarize([r for r in allrecs if r["config"] == c["name"]])
                             for c in configs}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n=== grader calibration ({args.served_model}, n={len(rows)}) ===")
    print(f"{'config':12s} {'cov':>5s} {'acc4':>6s} {'MAE%':>6s} {'pearson':>8s} "
          f"{'c_tok':>7s} {'tools':>6s}")
    for name, s in summary["by_config"].items():
        if s.get("parsed"):
            print(f"{name:12s} {s['coverage']:>5} {s['accuracy_4cat']:>6} {s['mae_pct']:>6} "
                  f"{str(s['pearson']):>8} {s['mean_completion_tokens']:>7} {s['mean_tool_calls']:>6}")
        else:
            print(f"{name:12s}  no parsed grades")
    print(f"\n[done] -> {out_dir/'summary.json'}  (golden MAE floor = 3.9%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="gradingbench.csv")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--served-model", default="deepseek-v4-flash")
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--n", type=int, default=200, help="stratified subset size")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--max-turns", type=int, default=6, help="pytool grader loop cap")
    ap.add_argument("--concurrency", type=int, default=200)
    ap.add_argument("--configs", default=",".join(c["name"] for c in CONFIGS))
    ap.add_argument("--limit", type=int, default=None, help="smoke: cap rows after sampling")
    ap.add_argument("--run-id", default="grader_calibration")
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
