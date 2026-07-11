"""Grade proof candidates with the CALIBRATED flash grader (high_notool), async.

Production scorer that mirrors `calibrate_grader.py`'s validated `high_notool` config
exactly — the one whose aggregate Pearson ~0.87 we trust (results/grader_calibration.md):

  AsyncChatClient.chat_raw(reasoning="high", max_tokens=65536, tools=None)
  prompt = prompts/grader.md (paper B.5) with problem_statement/solution/guidelines/student_answer
  score  = grader.parse_score(<points>N out of 7</points>)

NOT the legacy sync grader.py (reasoning="default", max_tokens=8192 — truncates to None).
chat_raw (identical request to chat for tools=None) lets us also archive the grader's
reasoning_content alongside its visible rationale.

Grades EVERY candidate of each run (k>=1) `--passes` times. Empty proofs (runaway /
incomplete) are NOT sent to the grader — they are recorded as score=0 directly. Writes one
lossless record per (run, pid, candidate, pass) to runs/<run_id>/<out_name>, append+resume.
Optional --ids-file restricts to a problem-id subset; default grades all problems.

Example (4 configs x 60 x k=4, 2 passes, skip-empty, 1000-way):
  python grade_proofs.py \
    --run-ids dsv4-flash__high_notool_k4,dsv4-flash__high_pytool_k4,dsv4-flash__max_notool_k4,dsv4-flash__max_pytool_k4 \
    --data ../data/proofbench_v2.csv --passes 2 \
    --base-url https://api.deepseek.com/v1 --served-model deepseek-v4-flash \
    --api-key-env DEEPSEEK_API_KEY --reasoning high --max-tokens 65536 \
    --concurrency 1000 --out-name grades_flashHighNotool_k4_2pass.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
import time

import pandas as pd
from openai import AsyncOpenAI

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from grader import parse_score  # noqa: E402  (canonical <points> parser)


def _reasoning_extra(reasoning: str) -> dict:
    """DeepSeek reasoning control via extra_body. high/max -> reasoning_effort; no_think ->
    thinking disabled; default -> nothing. We never send temperature/top_p (thinking mode
    ignores them); the calibrated production grader is 'high'."""
    if reasoning in ("high", "max"):
        return {"reasoning_effort": reasoning}
    if reasoning == "no_think":
        return {"thinking": {"type": "disabled"}}
    return {}


def load_run(run_id: str) -> list[dict]:
    """Every candidate of a run, flattened: one dict per (problem, candidate_idx)."""
    path = EVAL_ROOT / "runs" / run_id / "responses.jsonl"
    out: list[dict] = []
    for line in path.open():
        r = json.loads(line)
        for j, c in enumerate(r["candidates"]):
            out.append({"pid": r["problem_id"], "cand": j, "subset": r["subset"],
                        "category": r["category"], "level": r["level"],
                        "problem": r["problem"],
                        # grade the meta-stripped proof body; fall back to raw text
                        "text": c.get("graded_text") or c.get("text") or ""})
    return out


async def amain(args) -> None:
    key = os.environ.get(args.api_key_env)
    if not key:
        sys.exit(f"empty {args.api_key_env}")
    run_ids = [r.strip() for r in args.run_ids.split(",") if r.strip()]
    tpl = (EVAL_ROOT / "prompts" / "grader.md").read_text()
    src = pd.read_csv(args.data).set_index("Problem ID")
    keep = set(json.loads(Path(args.ids_file).read_text())) if args.ids_file else None

    # optional targeted (template, problem) selection for smoke: --pairs t0:PB-Basic-001,...
    pair_set = None
    if args.pairs:
        pair_set = set()
        for tok in args.pairs.split(","):
            tid, pid = tok.strip().split(":")
            rid = next((r for r in run_ids if f"__{tid}__" in r or r == tid), None)
            if rid is None:
                sys.exit(f"--pairs: no run-id matching template '{tid}' in {run_ids}")
            pair_set.add((rid, pid))

    cands = {r: load_run(r) for r in run_ids}
    out_paths = {r: EVAL_ROOT / "runs" / r / args.out_name for r in run_ids}

    # resume: which (run, pid, cand, pass) already written
    done: set[tuple] = set()
    for r in run_ids:
        if out_paths[r].exists():
            for line in out_paths[r].open():
                d = json.loads(line)
                done.add((r, d["problem_id"], d["candidate_idx"], d["pass"]))
    if done:
        print(f"[resume] {len(done)} records already present")

    files = {r: out_paths[r].open("a") for r in run_ids}

    def write(run_id, rec):
        files[run_id].write(json.dumps(rec, ensure_ascii=False) + "\n")
        files[run_id].flush()

    # split work: empties recorded as score=0 directly; non-empty -> grader tasks
    tasks, n_empty, n_skip = [], 0, 0
    for r in run_ids:
        for c in cands[r]:
            if keep is not None and c["pid"] not in keep:
                continue
            if pair_set is not None and (r, c["pid"]) not in pair_set:
                continue
            for p in range(args.passes):
                if (r, c["pid"], c["cand"], p) in done:
                    continue
                if not c["text"].strip():
                    write(r, {"run_id": r, "problem_id": c["pid"], "candidate_idx": c["cand"],
                              "pass": p, "subset": c["subset"], "category": c["category"],
                              "level": c["level"], "score": 0, "rationale": "empty proof (runaway/incomplete)",
                              "finish_reason": "empty", "grader_model": args.served_model,
                              "grader_config": f"{args.reasoning}_notool"})
                    n_empty += 1
                else:
                    tasks.append((r, c, p))
    for f in files.values():
        f.flush()
    print(f"[grade] runs={len(run_ids)} | empties recorded as 0: {n_empty} | grader calls: "
          f"{len(tasks)} | passes={args.passes} | conc={args.concurrency}")

    client = AsyncOpenAI(base_url=args.base_url, api_key=key, max_retries=5, timeout=3600.0)
    sema = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    counter = {"done": 0, "total": len(tasks)}

    async def work(run_id, c, p):
        async with sema:
            row = src.loc[c["pid"]]
            prompt = tpl.format(problem_statement=c["problem"], solution=row["Solution"],
                                guidelines=row["Grading guidelines"], student_answer=c["text"])
            try:
                t0 = time.monotonic()
                resp = await client.chat.completions.create(
                    model=args.served_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=args.max_tokens,
                    extra_body=_reasoning_extra(args.reasoning))
                latency = round(time.monotonic() - t0, 2)
                m = resp.choices[0].message
                content = m.content or ""
                g = parse_score(content)
                u = resp.usage
                rtoks = getattr(u.completion_tokens_details, "reasoning_tokens", None) \
                    if u and u.completion_tokens_details else None
                rec = {"run_id": run_id, "problem_id": c["pid"], "candidate_idx": c["cand"],
                       "pass": p, "subset": c["subset"], "category": c["category"],
                       "level": c["level"], "score": g["score"], "rationale": g["rationale"],
                       "grader_content": content,
                       "grader_reasoning": (m.model_extra or {}).get("reasoning_content") or "",
                       "finish_reason": resp.choices[0].finish_reason,
                       "completion_tokens": u.completion_tokens if u else None,
                       "reasoning_tokens": rtoks, "usage": u.model_dump() if u else None,
                       "latency_s": latency,
                       "grader_model": args.served_model, "grader_config": f"{args.reasoning}_notool"}
            except Exception as e:  # noqa: BLE001 - keep batch alive
                rec = {"run_id": run_id, "problem_id": c["pid"], "candidate_idx": c["cand"],
                       "pass": p, "subset": c["subset"], "category": c["category"],
                       "level": c["level"], "score": None, "rationale": f"error: {e}",
                       "finish_reason": "error", "grader_model": args.served_model,
                       "grader_config": f"{args.reasoning}_notool"}
        async with lock:
            write(run_id, rec)
            counter["done"] += 1
            if counter["done"] % 50 == 0 or counter["done"] == counter["total"]:
                print(f"  [{counter['done']}/{counter['total']}] last {c['pid']} #{c['cand']} p{p}: score={rec['score']}")

    if tasks:
        await asyncio.gather(*[work(r, c, p) for r, c, p in tasks])
    for f in files.values():
        f.close()
    await client.close()

    aggregate(run_ids, out_paths, args)


def _agg(scores: list[float]) -> dict:
    n = len(scores)
    if not n:
        return {"n": 0, "mean": None, "almost+": None, "correct": None}
    return {"n": n, "mean": round(mean(scores), 3),
            "almost+": round(sum(s >= 6 for s in scores) / n, 3),
            "correct": round(sum(s >= 7 for s in scores) / n, 3)}


def aggregate(run_ids, out_paths, args) -> None:
    """best-of-k (per-problem max over candidates) and mean-of-k, overall + by cut + deltas.
    A candidate's score = mean over its passes (empties=0). Per-problem best/mean over the k."""
    per_run = {}
    best_by_pid = {}
    for r in run_ids:
        recs = [json.loads(l) for l in out_paths[r].open()]
        # (pid,cand) -> [pass scores]; meta per pid
        cs = defaultdict(list)
        meta = {}
        for d in recs:
            if d["score"] is not None:
                cs[(d["problem_id"], d["candidate_idx"])].append(d["score"])
            meta[d["problem_id"]] = d
        # per-candidate mean over passes
        cand_score = {k: mean(v) for k, v in cs.items() if v}
        by_pid = defaultdict(list)
        for (pid, cand), s in cand_score.items():
            by_pid[pid].append(s)
        best = {pid: max(v) for pid, v in by_pid.items()}
        mof = {pid: mean(v) for pid, v in by_pid.items()}
        best_by_pid[r] = best
        # pass agreement: candidates graded with >=2 passes that agree exactly
        twin = [v for v in cs.values() if len(v) >= 2]
        agree = round(sum(v[0] == v[1] for v in twin) / len(twin), 3) if twin else None
        cut = {}
        for field in ("subset", "level", "category"):
            g = defaultdict(list)
            for pid, b in best.items():
                g[meta[pid][field]].append(b)
            cut[field] = {k: _agg(v) for k, v in sorted(g.items())}
        per_run[r] = {"n_problems": len(by_pid), "n_candidates_scored": len(cand_score),
                      "best_of_k": _agg(list(best.values())),
                      "mean_of_k": _agg(list(mof.values())),
                      "pass_exact_agreement": agree,
                      "by_subset": cut["subset"], "by_level": cut["level"],
                      "by_category": cut["category"],
                      "best_per_problem": {p: round(v, 2) for p, v in sorted(best.items())}}

    out = {"grader": f"{args.served_model} {args.reasoning}_notool", "passes": args.passes,
           "runs": per_run, "deltas": {}}
    # pairwise deltas on best-of-k (tool: pytool-notool; reasoning: max-high)
    def delta(a, b):
        pa, pb = best_by_pid.get(a, {}), best_by_pid.get(b, {})
        common = sorted(set(pa) & set(pb))
        d = [pb[p] - pa[p] for p in common]
        return {"ref": a, "vs": b, "n": len(common),
                "mean_delta": round(mean(d), 3) if d else None,
                "vs_better": sum(x > 0 for x in d), "ref_better": sum(x < 0 for x in d),
                "tie": sum(x == 0 for x in d)} if d else None
    pairs = {"tool@high": ("dsv4-flash__high_notool_k4", "dsv4-flash__high_pytool_k4"),
             "tool@max": ("dsv4-flash__max_notool_k4", "dsv4-flash__max_pytool_k4"),
             "reasoning@notool": ("dsv4-flash__high_notool_k4", "dsv4-flash__max_notool_k4"),
             "reasoning@pytool": ("dsv4-flash__high_pytool_k4", "dsv4-flash__max_pytool_k4")}
    for name, (a, b) in pairs.items():
        if a in best_by_pid and b in best_by_pid:
            out["deltas"][name] = delta(a, b)

    out_dir = EVAL_ROOT / "runs" / f"_grade_{args.reasoning}_notool_k4"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("\n=== grading summary (best-of-k | mean-of-k) ===")
    for r in run_ids:
        s = per_run[r]
        b, m = s["best_of_k"], s["mean_of_k"]
        print(f"{r}:")
        print(f"   best-of-k mean={b['mean']} almost+={b['almost+']} correct={b['correct']}  "
              f"| mean-of-k mean={m['mean']} correct={m['correct']}  "
              f"(probs={s['n_problems']}, pass-agree={s['pass_exact_agreement']})")
        print(f"   by subset: " + "  ".join(f"{k}={v['mean']}" for k, v in s["by_subset"].items()))
    print("\n--- deltas on best-of-k ---")
    for name, d in out["deltas"].items():
        if d:
            print(f"   {name:18s} ({d['vs']} - {d['ref']}): mean={d['mean_delta']} "
                  f"vs_better={d['vs_better']} ref_better={d['ref_better']} tie={d['tie']}")
    print(f"\n[done] -> {out_dir/'summary.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-ids", required=True, help="comma-separated run ids to grade")
    ap.add_argument("--data", required=True, help="source CSV (Solution + Grading guidelines)")
    ap.add_argument("--ids-file", default=None, help="JSON list of problem ids to restrict to")
    ap.add_argument("--pairs", default=None,
                    help="targeted smoke: tid:pid,tid:pid,... — grade only these template×problem cells")
    ap.add_argument("--passes", type=int, default=2, help="grader calls per candidate")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--served-model", default="deepseek-v4-flash")
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--reasoning", default="high", choices=["default", "no_think", "high", "max"])
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--concurrency", type=int, default=200)
    ap.add_argument("--out-name", default="grades_flashHighNotool_k4_2pass.jsonl")
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
