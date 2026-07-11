"""Batch driver for DSMV2-Simple-3R distill-data generation.

Runs the prove->verify->rank->refine->select pipeline over a problem parquet, with cross-problem
concurrency, and writes ONE full-trace record per problem (every stage call's prompt + reasoning
+ content preserved) to outputs/<run_id>/records.jsonl. Lossless append + resume (skip done
problem_ids), matching distill_gen/collect.py conventions and the host's bit-flip instability.

Two concurrency knobs:
  --concurrency           global in-flight API calls (shared by all stages of all problems)
  --problem-concurrency   how many problems are mid-pipeline at once

Usage (smoke):
    DEEPSEEK_API_KEY=... uv run python distill_gen/math_3r/run.py \
        --input distill_gen/math_3r/random16.parquet --run-id r3_smoke16
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "evaluation" / "harness"))

from async_client import AsyncChatClient            # noqa: E402
from pipeline import Engine, solve_problem           # noqa: E402

BASE_URL = "https://api.deepseek.com/v1"
MODEL = "deepseek-v4-flash"
PROMPT_DIR = HERE / "prompts"
DEFAULT_INPUT = HERE / "random16.parquet"
OUT_ROOT = HERE / "outputs"


def pid_of(problem: str) -> str:
    return hashlib.blake2b(problem.encode(), digest_size=8).hexdigest()


def done_keys(records_path: Path) -> set[str]:
    keys: set[str] = set()
    if not records_path.exists():
        return keys
    with open(records_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                keys.add(json.loads(line)["problem_id"])
            except json.JSONDecodeError:
                continue  # tolerate a torn last line from an interrupted run
    return keys


def prompt_shas() -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(PROMPT_DIR.iterdir()) if p.is_file()}


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--run-id", default="r3")
    ap.add_argument("--model", default=MODEL, help="DeepSeek model id (e.g. deepseek-v4-flash / deepseek-v4-pro)")
    ap.add_argument("--max-tokens", type=int, default=180000, help="per-call budget, all stages")
    ap.add_argument("--effort", default="high", choices=["high", "max"])
    ap.add_argument("--num-provers", type=int, default=6)
    ap.add_argument("--verify-k", type=int, default=2, help="verifiers per valid proof")
    ap.add_argument("--num-refiners", type=int, default=3)
    ap.add_argument("--num-selectors", type=int, default=4, help="selector voters (majority vote)")
    ap.add_argument("--concurrency", type=int, default=48, help="global in-flight API calls")
    ap.add_argument("--problem-concurrency", type=int, default=32, help="problems mid-pipeline")
    ap.add_argument("--max-retries", type=int, default=1,
                    help="attempts per call (1 = NO retry; failed/parse-error samples are dropped)")
    ap.add_argument("--limit", type=int, default=0, help="cap #problems (0 = all)")
    args = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("set DEEPSEEK_API_KEY in the environment (never hard-code it)")

    cols = ["problem", "origin", "category", "competition", "source", "nm_uuid"]
    # carry difficulty annotation through when the input has it (e.g. hard2000.parquet) so the
    # records are self-contained; inputs without those columns (random16/500) just omit them.
    avail = set(pq.ParquetFile(args.input).schema_arrow.names)
    cols += [c for c in ("difficulty_source", "difficulty_value") if c in avail]
    rows = pq.read_table(args.input, columns=cols).to_pylist()
    if args.limit:
        rows = rows[: args.limit]

    out_dir = OUT_ROOT / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    done = done_keys(records_path)
    todo = [r for r in rows if pid_of(r["problem"]) not in done]
    print(f"[r3] problems={len(rows)} done={len(done)} todo={len(todo)} "
          f"| provers={args.num_provers} effort={args.effort} max_tokens={args.max_tokens}",
          flush=True)

    (out_dir / "run_meta.json").write_text(json.dumps({
        "run_id": args.run_id, "model": args.model, "base_url": BASE_URL, "effort": args.effort,
        "max_tokens": args.max_tokens, "num_provers": args.num_provers,
        "verify_k": args.verify_k, "num_refiners": args.num_refiners,
        "num_selectors": args.num_selectors,
        "concurrency": args.concurrency, "problem_concurrency": args.problem_concurrency,
        "max_retries": args.max_retries, "input": str(args.input), "n_problems": len(rows),
        "prompt_sha256": prompt_shas(), "git_commit": git_commit(),
    }, indent=2, ensure_ascii=False))

    if not todo:
        print("[r3] nothing to do (all done).")
        return

    client = AsyncChatClient(BASE_URL, args.model, api_key=key,
                             max_connections=args.concurrency + 8, max_retries=args.max_retries)
    call_sem = asyncio.Semaphore(args.concurrency)
    prob_sem = asyncio.Semaphore(args.problem_concurrency)
    engine = Engine(client, call_sem, max_tokens=args.max_tokens, effort=args.effort)
    fout = open(records_path, "a")
    write_lock = asyncio.Lock()
    state = {"done": 0, "ctok": 0, "rtok": 0, "calls": 0, "errs": 0,
             "final_fallback": 0, "t0": time.monotonic()}

    async def worker(row: dict) -> None:
        async with prob_sem:
            pid = pid_of(row["problem"])
            trace = await solve_problem(row["problem"], engine, num_provers=args.num_provers,
                                        verify_k=args.verify_k, num_refiners=args.num_refiners,
                                        num_selectors=args.num_selectors)
            rec = {
                "problem_id": pid, "problem": row["problem"], "origin": row["origin"],
                "category": row["category"], "competition": row["competition"],
                "source": row["source"], "nm_uuid": row["nm_uuid"],
                "difficulty_source": row.get("difficulty_source"),
                "difficulty_value": row.get("difficulty_value"),
                "run_id": args.run_id, "model": args.model, "effort": args.effort,
                "max_tokens": args.max_tokens, "num_provers": args.num_provers,
                "verify_k": args.verify_k, "num_refiners": args.num_refiners,
                "num_selectors": args.num_selectors,
                "final_proof": trace["final_proof"], "final_source": trace["final_source"],
                "selected_id": trace.get("selected_id"), "selected_ids": trace.get("selected_ids"),
                "counts": trace["counts"], "totals": trace["totals"], "stages": trace["stages"],
            }
            async with write_lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                t = trace["totals"]
                state["done"] += 1
                state["ctok"] += t["completion_tokens"]
                state["rtok"] += t["reasoning_tokens"]
                state["calls"] += t["n_calls"]
                state["errs"] += t["n_errors"]
                state["final_fallback"] += int(not trace["final_source"].startswith("select"))
                n = state["done"]
                dt = time.monotonic() - state["t0"]
                print(f"  [{n}/{len(todo)}] {row['problem'][:40]!r} valid={trace['counts']['n_valid_proofs']}"
                      f"/{args.num_provers} refined={trace['counts']['n_refined_valid']} "
                      f"src={trace['final_source']} | calls={state['calls']} errs={state['errs']} "
                      f"{state['ctok']/1e6:.1f}M ctok | {n/dt*60:.1f} prob/min", flush=True)

    try:
        await asyncio.gather(*(worker(r) for r in todo))
    finally:
        fout.close()
        await client.aclose()
    print(f"[r3] finished: {state['done']} problems, {state['calls']} calls, {state['errs']} errored, "
          f"{state['final_fallback']} used fallback | {state['ctok']/1e6:.1f}M ctok "
          f"({state['rtok']/1e6:.1f}M reasoning) -> {records_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
