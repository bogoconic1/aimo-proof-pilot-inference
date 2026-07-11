"""Local-SGLang prompt-template sweep for IMO-ProofBench (single-round, no-tool).

Adapted from evaluation/harness/run_template_sweep.py. Same template set, same
problem CSV, and — critically — the SAME responses.jsonl schema and run-dir layout,
so the existing evaluation/harness/{extract_proof,grade_proofs,analyze_sweep}.py can
grade these runs later (against the DeepSeek API) with zero changes.

Differences vs the DeepSeek version:
  * Points at a LOCAL SGLang OpenAI-compatible endpoint (--base-url).
  * Sends temperature / top_p (OLMo reasoning models loop at T=0; DeepSeek thinking
    mode ignored sampling, these models do not). reasoning_effort is NOT sent unless
    --send-reasoning-effort is given.
  * Robust final-answer extraction: if the served reasoning parser already split the
    chain-of-thought into reasoning_content, content is the final proof. If it did NOT
    (content still holds a `</think>` block), we split on the last `</think>` so `text`
    is always the post-thinking proof and the CoT goes to reasoning_content. This keeps
    `text` (what the grader reads) = the final proof, matching the DeepSeek runs.

Run dirs: runs/<model-label>__<tN>__<reasoning>_notool/  (under --runs-root).
Crash-safe: each candidate is appended to candidates_raw.jsonl and resumed from it;
responses.jsonl is regrouped at the end in CSV order.

Example (one model, all 8 templates, 60 problems, k=3, against a local server on :30000):
  OPENAI_API_KEY=EMPTY uv run python gen_local.py \
    --data ../../evaluation/data/proofbench_v2.csv \
    --templates-json ../../evaluation/data/imo_proofbench_single_round_prompt_templates.json \
    --model-label stage1-v2-7b --served-model stage1-v2-7b \
    --base-url http://127.0.0.1:30000/v1 --api-key-env OPENAI_API_KEY \
    --max-tokens 60000 --temperature 0.6 --top-p 0.95 --k 3 --concurrency 16 \
    --runs-root ./evaluation_local/runs
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


def repro_info() -> dict:
    def _git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=str(HERE),
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:  # noqa: BLE001
            return None
    sha = hashlib.sha256((HERE / "gen_local.py").read_bytes()).hexdigest()[:16]
    return {"git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "harness_sha256": {"gen_local.py": sha}}


def subset_of(pid: str) -> str:
    return "advanced" if "Advanced" in pid else "basic"


def short_id(tid: str) -> str:
    return tid.split("_", 1)[0]


def load_templates(path: Path, which: str | None) -> list[dict]:
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


def split_final(content: str, reasoning: str) -> tuple[str, str]:
    """Return (final_proof_text, reasoning). If the server's reasoning parser already
    separated the CoT, `content` is the proof and we keep it. If not, `content` still
    contains a `<think>...</think>` block — split on the LAST `</think>` so `text` is the
    post-thinking proof and the thinking joins reasoning_content."""
    content = content or ""
    reasoning = reasoning or ""
    if "</think>" in content:
        pre, _, post = content.rpartition("</think>")
        pre = pre.replace("<think>", "")
        reasoning = (reasoning + "\n" + pre).strip() if reasoning else pre.strip()
        content = post.strip()
    return content, reasoning


async def amain(args) -> None:
    api_key = os.environ.get(args.api_key_env) or "EMPTY"  # local server ignores it

    templates = load_templates(Path(args.templates_json), args.templates)
    df = pd.read_csv(args.data)
    if args.limit:
        df = df.head(args.limit)
    order = list(df["Problem ID"])
    rows = {r["Problem ID"]: r for _, r in df.iterrows()}

    runs_root = Path(args.runs_root)
    run_ids, out_dirs, raw_paths, done, raw_f = {}, {}, {}, {}, {}
    for t in templates:
        sid = short_id(t["id"])
        rid = f"{args.model_label}__{sid}__{args.reasoning}_notool"
        out_dir = runs_root / rid
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = out_dir / "candidates_raw.jsonl"
        run_ids[sid], out_dirs[sid], raw_paths[sid] = rid, out_dir, raw_path

        meta_path = out_dir / "run_meta.json"
        if not (args.redo_empties and meta_path.exists()):
            meta_path.write_text(json.dumps({
                "model_name": args.model_label, "served_model": args.served_model,
                "base_url": args.base_url, "condition": "notool", "data": str(args.data),
                "reasoning": args.reasoning, "max_tokens": args.max_tokens, "k": args.k,
                "temperature": args.temperature, "top_p": args.top_p,
                "send_reasoning_effort": args.send_reasoning_effort,
                "concurrency": args.concurrency,
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
        if args.redo_empties:
            d = {k: v for k, v in d.items() if v.get("failure_type") != "empty"}
        done[sid] = d
        raw_f[sid] = raw_path.open("a")

    tasks = [(t, pid, j) for t in templates for pid in order for j in range(args.k)
             if (pid, j) not in done[short_id(t["id"])]]
    total = len(templates) * len(order) * args.k
    n_done = total - len(tasks)
    print(f"[run] {args.model_label}: {len(templates)} templates x {len(order)} problems "
          f"x k={args.k} = {total} candidates | {n_done} present | {len(tasks)} to do | "
          f"conc={args.concurrency} max_tok={args.max_tokens} T={args.temperature}",
          flush=True)

    client = AsyncOpenAI(base_url=args.base_url, api_key=api_key,
                         max_retries=args.max_retries, timeout=args.timeout)
    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    counter = {"done": n_done, "fail": 0}

    async def work(t, pid, j):
        sid = short_id(t["id"])
        row = rows[pid]
        sys_p, user_p = t["system_prompt"], render_user(t, row["Problem"])
        kwargs = dict(model=args.served_model,
                      messages=[{"role": "system", "content": sys_p},
                                {"role": "user", "content": user_p}],
                      max_tokens=args.max_tokens)
        if args.temperature is not None:
            kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            kwargs["top_p"] = args.top_p
        if args.send_reasoning_effort:
            kwargs["extra_body"] = {"reasoning_effort": args.reasoning}
        async with sem:
            t0 = time.monotonic()
            try:
                resp = await client.chat.completions.create(**kwargs)
            except Exception as e:  # noqa: BLE001 - log + skip; resume retries next run
                async with write_lock:
                    counter["fail"] += 1
                    print(f"  [FAIL] {sid} {pid} c{j}: {e!r}", flush=True)
                return None
            latency = round(time.monotonic() - t0, 2)

        msg = resp.choices[0].message
        finish = resp.choices[0].finish_reason
        raw_content = msg.content or ""
        raw_reasoning = (msg.model_extra or {}).get("reasoning_content") or ""
        text, reasoning = split_final(raw_content, raw_reasoning)
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
            "max_tokens": args.max_tokens,
            "usage": u.model_dump() if u else None,
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
                  f"{rec['completion_tokens']} tok, {latency}s, "
                  f"{finish}/{rec['failure_type']} (text {len(text)}c)", flush=True)
        return rec

    if tasks:
        await asyncio.gather(*[work(t, pid, j) for (t, pid, j) in tasks])
    for f in raw_f.values():
        f.close()
    await client.close()

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
              f"{sum(1 for pid in order if (pid,0) in d)}/{len(order)} problems", flush=True)

    print(f"\n[done] {args.model_label}: {counter['done']}/{total} candidates | "
          f"failures (retry on rerun): {counter['fail']}", flush=True)
    print("[runs] " + ", ".join(run_ids[short_id(t['id'])] for t in templates), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--templates-json", required=True)
    ap.add_argument("--templates", default=None,
                    help="comma list of short ids (t0,t1) or full ids; default = all")
    ap.add_argument("--base-url", required=True, help="local SGLang OpenAI endpoint")
    ap.add_argument("--model-label", required=True, help="label used in run dir")
    ap.add_argument("--served-model", required=True, help="SGLang served-model-name / API model id")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--reasoning", default="high", help="label only (used in run dir name)")
    ap.add_argument("--send-reasoning-effort", action="store_true",
                    help="also send extra_body.reasoning_effort (DeepSeek-only; off for OLMo)")
    ap.add_argument("--max-tokens", type=int, default=60000,
                    help="combined reasoning+output budget (cap to model context - prompt)")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--k", type=int, default=3, help="samples per problem")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=7200.0)
    ap.add_argument("--limit", type=int, default=None, help="first N problems (smoke)")
    ap.add_argument("--redo-empties", action="store_true",
                    help="re-generate ONLY prior-empty (runaway) candidates")
    ap.add_argument("--runs-root", required=True, help="dir to write runs/<rid>/ under")
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
