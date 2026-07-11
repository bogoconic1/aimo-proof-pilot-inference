"""Prompt-template sweep for IMO-ProofBench, single-round, no-tool.

Generates proof candidates for every (template x problem x sample) with DeepSeek V4
Flash via the official AsyncOpenAI SDK pointed at the DeepSeek endpoint. Each template
in data/imo_proofbench_single_round_prompt_templates.json is a (system_prompt,
user_prompt_template) pair; we send messages=[system, user(problem)] and DO NOT send
temperature/top_p (DeepSeek thinking mode ignores them). reasoning_effort goes through
extra_body. Condition is fixed: high reasoning, no tool, k samples per problem.

One run dir per template (runs/<model>__<tN>__<reasoning>_notool/) so the existing
grade_proofs.py can grade them by run-id with the fixed flash high_notool grader.
For crash safety each finished candidate is appended to candidates_raw.jsonl and the
run resumes from it; responses.jsonl is regrouped at the end in CSV order.

Example (full sweep, all 8 templates, 60 problems, k=3, 48-way):
  DEEPSEEK_API_KEY=... uv run python run_template_sweep.py \
    --data ../data/proofbench_v2.csv \
    --templates-json ../data/imo_proofbench_single_round_prompt_templates.json \
    --model-name dsv4-flash --served-model deepseek-v4-flash \
    --base-url https://api.deepseek.com/v1 --api-key-env DEEPSEEK_API_KEY \
    --reasoning high --max-tokens 131072 --k 3 --concurrency 48

Smoke (t0,t1 x 4 problems x k=1):
  DEEPSEEK_API_KEY=... uv run python run_template_sweep.py ... \
    --templates t0,t1 --limit 4 --k 1 --concurrency 8
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent


def repro_info() -> dict:
    """Pin the code version: git commit + dirty flag + sha256 of this script, so an
    uncommitted run is still identifiable."""
    def _git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=str(HERE),
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:  # noqa: BLE001
            return None
    sha = hashlib.sha256((HERE / "run_template_sweep.py").read_bytes()).hexdigest()[:16]
    return {"git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "harness_sha256": {"run_template_sweep.py": sha}}


def subset_of(pid: str) -> str:
    return "advanced" if "Advanced" in pid else "basic"


def short_id(tid: str) -> str:
    """'t0_minimal_rigorous_baseline' -> 't0'."""
    return tid.split("_", 1)[0]


def load_templates(path: Path, which: str | None) -> list[dict]:
    """Return the selected templates in file order. `which` is a comma list of short
    ids ('t0,t1') or full ids; None = all."""
    data = json.loads(path.read_text())
    tpls = data["templates"]
    if not which:
        return tpls
    want = {w.strip() for w in which.split(",") if w.strip()}
    sel = [t for t in tpls if short_id(t["id"]) in want or t["id"] in want]
    missing = want - {short_id(t["id"]) for t in tpls} - {t["id"] for t in tpls}
    if missing:
        sys.exit(f"--templates: unknown ids {sorted(missing)}; "
                 f"available: {[short_id(t['id']) for t in tpls]}")
    return sel


_STATUS_PATS = [
    ("complete", re.compile(r"\b(complete solution|complete proof|fully correct|"
                            r"verdict[:\s]+complete|status[:\s]+complete)\b", re.I)),
    ("almost", re.compile(r"\balmost complete\b", re.I)),
    ("partial", re.compile(r"\b(partial (?:solution|progress|result)|"
                           r"verdict[:\s]+partial|status[:\s]+partial|"
                           r"mark(?:ed)? .{0,20}partial)\b", re.I)),
]


def claimed_status(text: str) -> str | None:
    """Best-effort: what completeness did the model claim? Heuristic only (full content
    is saved, so this can be re-derived/refined at analysis time)."""
    if not text:
        return None
    head = text[:1500]
    for label, pat in _STATUS_PATS:
        if pat.search(head):
            return label
    return None


def failure_type(text: str, finish_reason: str | None) -> str:
    if not (text or "").strip():
        return "empty"          # runaway / no final content
    if finish_reason == "length":
        return "truncated"      # has partial content but hit the cap
    return "ok"


def render_user(tpl: dict, problem: str) -> str:
    # str.replace, NOT .format — templates/problems contain LaTeX braces.
    return tpl["user_prompt_template"].replace("{problem}", problem)


async def amain(args) -> None:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        sys.exit(f"--api-key-env {args.api_key_env} is empty")

    templates = load_templates(Path(args.templates_json), args.templates)
    df = pd.read_csv(args.data)
    if args.limit:
        df = df.head(args.limit)
    order = list(df["Problem ID"])
    rows = {r["Problem ID"]: r for _, r in df.iterrows()}

    # per-template run dirs + resume state + append handles
    run_ids, out_dirs, raw_paths, done, raw_f = {}, {}, {}, {}, {}
    for t in templates:
        sid = short_id(t["id"])
        rid = f"{args.model_name}__{sid}__{args.reasoning}_notool"
        out_dir = EVAL_ROOT / "runs" / rid
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = out_dir / "candidates_raw.jsonl"
        run_ids[sid], out_dirs[sid], raw_paths[sid] = rid, out_dir, raw_path

        # keep the original meta (and its max_tokens) intact on a --redo-empties pass;
        # per-candidate max_tokens (below) is the authoritative budget record
        meta_path = out_dir / "run_meta.json"
        if not (args.redo_empties and meta_path.exists()):
            meta_path.write_text(json.dumps({
                "model_name": args.model_name, "served_model": args.served_model,
                "base_url": args.base_url, "condition": "notool", "data": str(args.data),
                "reasoning": args.reasoning, "max_tokens": args.max_tokens, "k": args.k,
                "concurrency": args.concurrency, "no_sampling_params": True,
                "template_id": t["id"], "template_name": t.get("name"),
                "templates_json": str(args.templates_json),
                "system_prompt": t["system_prompt"],
                "user_prompt_template": t["user_prompt_template"],
                "repro": repro_info(),
            }, indent=2, ensure_ascii=False))

        d = {}
        if raw_path.exists():
            for line in raw_path.open():
                r = json.loads(line)
                d[(r["problem_id"], r["j"])] = r   # last write wins
        if args.redo_empties:  # re-attempt prior runaway/empty candidates
            d = {k: v for k, v in d.items() if v.get("failure_type") != "empty"}
        done[sid] = d
        raw_f[sid] = raw_path.open("a")

    tasks = [(t, pid, j) for t in templates for pid in order for j in range(args.k)
             if (pid, j) not in done[short_id(t["id"])]]
    total = len(templates) * len(order) * args.k
    n_done = total - len(tasks)
    print(f"[run] {len(templates)} templates x {len(order)} problems x k={args.k} = {total} "
          f"candidates | {n_done} already present | {len(tasks)} to do | conc={args.concurrency}")

    client = AsyncOpenAI(base_url=args.base_url, api_key=api_key,
                         max_retries=args.max_retries, timeout=args.timeout)
    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    counter = {"done": n_done, "fail": 0}

    async def work(t, pid, j):
        sid = short_id(t["id"])
        row = rows[pid]
        sys_p, user_p = t["system_prompt"], render_user(t, row["Problem"])
        async with sem:
            t0 = time.monotonic()
            try:
                resp = await client.chat.completions.create(
                    model=args.served_model,
                    messages=[{"role": "system", "content": sys_p},
                              {"role": "user", "content": user_p}],
                    max_tokens=args.max_tokens,
                    extra_body={"reasoning_effort": args.reasoning},
                )
            except Exception as e:  # noqa: BLE001 - log + skip; resume retries it next run
                async with write_lock:
                    counter["fail"] += 1
                    print(f"  [FAIL] {sid} {pid} c{j}: {e!r}")
                return None
            latency = round(time.monotonic() - t0, 2)

        msg = resp.choices[0].message
        finish = resp.choices[0].finish_reason
        text = msg.content or ""
        reasoning = (msg.model_extra or {}).get("reasoning_content") or ""
        u = resp.usage
        rtoks = getattr(u.completion_tokens_details, "reasoning_tokens", None) \
            if u and u.completion_tokens_details else None
        rec = {
            "template_id": t["id"], "problem_id": pid, "j": j, "subset": subset_of(pid),
            "category": row["Category"], "level": row["Level"], "problem": row["Problem"],
            "text": text, "finish_reason": finish,
            "failure_type": failure_type(text, finish),
            "claimed_status": claimed_status(text),
            "prompt_tokens": u.prompt_tokens if u else None,
            "completion_tokens": u.completion_tokens if u else None,
            "reasoning_tokens": rtoks, "latency_s": latency,
            "max_tokens": args.max_tokens,   # budget this candidate was generated under
            # lossless: the FULL usage object (incl total_tokens, prompt_cache_hit/miss
            # tokens, *_tokens_details) — the scalars above are convenience copies
            "usage": u.model_dump() if u else None,
            # lossless: full exchange incl the model's thinking
            "messages": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_p},
                {"role": "assistant", "content": text,
                 "reasoning_content": reasoning, "finish_reason": finish},
            ],
        }
        async with write_lock:
            f = raw_f[sid]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[sid][(pid, j)] = rec
            counter["done"] += 1
            print(f"  [{counter['done']}/{total}] {sid} {pid} c{j}: "
                  f"{rec['completion_tokens']} tok ({rtoks} rsn), {latency}s, "
                  f"{finish}/{rec['failure_type']}")
        return rec

    if tasks:
        await asyncio.gather(*[work(t, pid, j) for (t, pid, j) in tasks])
    for f in raw_f.values():
        f.close()
    await client.close()

    # regroup each template into responses.jsonl (grade_proofs.py-compatible schema)
    for t in templates:
        sid = short_id(t["id"])
        d = done[sid]
        with (out_dirs[sid] / "responses.jsonl").open("w") as f:
            for pid in order:
                cands = [d[(pid, j)] for j in range(args.k) if (pid, j) in d]
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
                                    "reasoning_tokens": c["reasoning_tokens"],
                                    "latency_s": c["latency_s"],
                                    "max_tokens": c.get("max_tokens"),
                                    "failure_type": c["failure_type"],
                                    "claimed_status": c["claimed_status"]} for c in cands],
                }, ensure_ascii=False) + "\n")
        print(f"[regroup] {run_ids[sid]}: "
              f"{sum(1 for pid in order if (pid,0) in d)}/{len(order)} problems")

    print(f"\n[done] {counter['done']}/{total} candidates | failures (will retry on rerun): "
          f"{counter['fail']}")
    print("[runs] " + ", ".join(run_ids[short_id(t['id'])] for t in templates))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--templates-json", required=True)
    ap.add_argument("--templates", default=None,
                    help="comma list of short ids (t0,t1) or full ids; default = all")
    ap.add_argument("--base-url", default="https://api.deepseek.com/v1")
    ap.add_argument("--model-name", default="dsv4-flash", help="label used in run dir")
    ap.add_argument("--served-model", default="deepseek-v4-flash", help="API model id")
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--reasoning", default="high", choices=["high", "max"],
                    help="reasoning_effort (thinking on; temperature/top_p not sent)")
    ap.add_argument("--max-tokens", type=int, default=131072,
                    help="combined reasoning+output budget")
    ap.add_argument("--k", type=int, default=3, help="samples per problem")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--limit", type=int, default=None, help="first N problems (smoke)")
    ap.add_argument("--redo-empties", action="store_true",
                    help="re-generate ONLY prior-empty (runaway) candidates with the "
                         "current --max-tokens; appends new results (last-write-wins)")
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
