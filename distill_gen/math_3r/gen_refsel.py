"""Generate EXTRA refine + select samples by full-random sampling from the existing proof pool.

Diversity-focused data augmentation: the expensive prove/verify stages are reused (read from the
per_problem pool), so we only pay for the cheap refine/select outputs. For each problem (the
hard2000 spine minus the too-easy ones) we draw fresh random candidate bundles:

  refine (N_REFINE per problem): pick 2-4 candidate proofs (that carry verifier reviews); each shown
    as its <solution> ONLY (self-evaluation stripped) + 1-2 random verifier reviews. -> refined proof.
  select (N_SELECT per problem): pick 2-4 candidates; each randomly presented as one of
    proof-only / proof+self-eval / proof+verification. -> selector returns <selected_id>.

Candidates come from the problem's math_3r prove proofs. Within each bundle they are relabeled
P0,P1,... (the selector parser accepts P#). Per-problem RNG is seeded from problem_id so a rerun
reproduces the same bundles. Output records mirror math_3r/run.py's schema (empty prove/verify,
populated refine/select) so pack_hf.py merges them by problem_id with no special-casing.

Too-easy problems (every scored proof verified perfect) are excluded, matching pack_hf --drop-easy.

Usage (smoke 2 problems, no full run):
    DEEPSEEK_API_KEY=... uv run python distill_gen/math_3r/gen_refsel.py \
        --pool /tmp/hf_dsflash_pack/data/per_problem.parquet --run-id r3_gen_refsel --limit 2
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "evaluation" / "harness"))

from async_client import AsyncChatClient                              # noqa: E402
from pipeline import Engine, _totals                                  # noqa: E402
from parser import (parse_refined_package, parse_selected_id,         # noqa: E402
                    _SOLUTION_RE, _SELFEVAL_RE)
from prompts import (render_refiner_prompt, render_selector_prompt,   # noqa: E402
                     to_messages)

BASE_URL = "https://api.deepseek.com/v1"
MODEL = "deepseek-v4-flash"
OUT_ROOT = HERE / "outputs"


def _extract(rx, text: str) -> str:
    m = rx.search(text or "")
    return (m.group(1).strip() if m else "")


def _fmt_score(s) -> str:
    return "?" if s is None else ("%g" % s)


def candidates_of(row: dict) -> list[dict]:
    """math_3r prove proofs for a problem -> {solution, self_eval, self_score, verifs[]}.
    solution = <solution> body only (self-eval stripped, per spec)."""
    out = []
    for p in json.loads(row["proofs_json"]):
        if p.get("pipeline") != "math_3r":
            continue
        sol = _extract(_SOLUTION_RE, p.get("content") or "")
        if not sol:
            continue
        verifs = [{"score": v.get("score"), "content": v.get("content") or ""}
                  for v in (p.get("verifications") or []) if (v.get("content") or "").strip()]
        out.append({"solution": sol, "self_eval": _extract(_SELFEVAL_RE, p.get("content") or ""),
                    "self_score": p.get("self_score"), "verifs": verifs,
                    "candidate_id": p.get("candidate_id"), "run_id": p.get("run_id")})
    return out


def _block(cid: str, parts_inner: list[str]) -> str:
    return "\n".join([f'<candidate id="{cid}">', *parts_inner, "</candidate>"])


def refine_bundle(rng: random.Random, cands: list[dict]) -> tuple[str, list[dict]]:
    """2-4 candidates, each = <solution> (no self-eval) + 1-2 random verifier reviews."""
    k = rng.randint(2, min(4, len(cands)))
    chosen = rng.sample(cands, k)
    blocks, meta = [], []
    for i, c in enumerate(chosen):
        cid = f"P{i}"
        nv = rng.randint(1, min(2, len(c["verifs"])))
        revs = rng.sample(c["verifs"], nv)
        inner = ["<proof>", c["solution"], "</proof>"]
        for v in revs:
            inner += [f'<verifier_review score="{_fmt_score(v["score"])}">', v["content"],
                      "</verifier_review>"]
        blocks.append(_block(cid, inner))
        meta.append({"bid": cid, "orig_id": c["candidate_id"], "src": c["run_id"],
                     "n_reviews": nv, "review_scores": [v["score"] for v in revs]})
    return "\n".join(blocks), meta


def select_bundle(rng: random.Random, cands: list[dict]) -> tuple[str, dict, list[dict]]:
    """2-4 candidates, each randomly proof-only / proof+self-eval / proof+verification."""
    k = rng.randint(2, min(4, len(cands)))
    chosen = rng.sample(cands, k)
    blocks, id_map, meta = [], {}, []
    for i, c in enumerate(chosen):
        cid = f"P{i}"
        id_map[cid] = c["solution"]
        forms = ["proof"]
        if c["self_eval"]:
            forms.append("proof_se")
        if c["verifs"]:
            forms.append("proof_v")
        form = rng.choice(forms)
        inner = ["<proof>", c["solution"], "</proof>"]
        vmean = None
        if form == "proof_se":
            inner += ["<self_evaluation>", c["self_eval"], "</self_evaluation>"]
        elif form == "proof_v":
            v = rng.choice(c["verifs"])
            inner += [f'<verifier_review score="{_fmt_score(v["score"])}">', v["content"],
                      "</verifier_review>"]
        scored = [v["score"] for v in c["verifs"] if v["score"] is not None]
        vmean = sum(scored) / len(scored) if scored else None
        blocks.append(_block(cid, inner))
        meta.append({"bid": cid, "orig_id": c["candidate_id"], "src": c["run_id"],
                     "form": form, "verifier_mean": vmean})
    return "\n".join(blocks), id_map, meta


def easy_pids(rows: list[dict]) -> set[str]:
    easy = set()
    for r in rows:
        means = []
        for p in json.loads(r["proofs_json"]):
            vs = [v["score"] for v in (p.get("verifications") or []) if v.get("score") is not None]
            if vs:
                means.append(sum(vs) / len(vs))
        if means and all(m == 1.0 for m in means):
            easy.add(r["problem_id"])
    return easy


def done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(ln)["problem_id"] for ln in open(path) if ln.strip()}


async def gen_problem(row: dict, engine: Engine, *, n_refine: int, n_select: int) -> dict | None:
    cands = candidates_of(row)
    refine_cands = [c for c in cands if c["verifs"]]
    if len(cands) < 2 or len(refine_cands) < 2:
        return None  # not enough material to form a bundle (rare on hard2000)
    rng = random.Random(int(row["problem_id"], 16))

    refine_specs = [refine_bundle(rng, refine_cands) for _ in range(n_refine)]
    select_specs = [select_bundle(rng, cands) for _ in range(n_select)]
    jobs = ([(to_messages(render_refiner_prompt(row["problem"], b)), f"refine/R{i}")
             for i, (b, _) in enumerate(refine_specs)]
            + [(to_messages(render_selector_prompt(row["problem"], b)), f"select/S{i}")
               for i, (b, _, _) in enumerate(select_specs)])
    calls = await engine.run_parallel(jobs)
    refine_calls, select_calls = calls[:n_refine], calls[n_refine:]

    refine_views = []
    for i, (c, (_, meta)) in enumerate(zip(refine_calls, refine_specs)):
        rp = parse_refined_package(c, f"R{i}")
        refine_views.append({"refiner_id": rp.refiner_id, "self_score": rp.self_score,
                             "valid": rp.valid, "bundle_candidates": meta, **c})
    votes, select_views = [], []
    for i, (c, (_, id_map, meta)) in enumerate(zip(select_calls, select_specs)):
        pick = parse_selected_id(c["content"])
        votes.append(pick if pick in id_map else None)
        select_views.append({"bundle_candidates": meta, "picked": pick, **c})

    stages = {"prove": [], "verify": [], "ranking": [], "refine": refine_views,
              "select": select_views}
    return {
        "problem_id": row["problem_id"], "problem": row["problem"], "origin": row["origin"],
        "category": row["category"], "competition": row["competition"], "source": row["source"],
        "nm_uuid": row["nm_uuid"], "difficulty_source": row["difficulty_source"],
        "difficulty_value": row["difficulty_value"],
        "run_id": engine_run_id, "model": MODEL, "effort": engine.effort,
        "max_tokens": engine.max_tokens, "num_provers": 0, "verify_k": 0,
        "num_refiners": n_refine, "num_selectors": n_select,
        "final_proof": "", "final_source": "gen_refsel", "selected_id": None,
        "selected_ids": votes,
        "counts": {"n_provers": 0, "n_valid_proofs": 0, "n_verifs": 0,
                   "n_refined_valid": sum(1 for v in refine_views if v["valid"])},
        "totals": _totals(stages), "stages": stages,
    }


engine_run_id = "r3_gen_refsel"  # set in main


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True, help="per_problem.parquet with proofs+verifs")
    ap.add_argument("--run-id", default="r3_gen_refsel")
    ap.add_argument("--n-refine", type=int, default=5)
    ap.add_argument("--n-select", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=180000)
    ap.add_argument("--effort", default="high", choices=["high", "max"])
    ap.add_argument("--concurrency", type=int, default=1200)
    ap.add_argument("--problem-concurrency", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    global engine_run_id
    engine_run_id = args.run_id
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("set DEEPSEEK_API_KEY in the environment (never hard-code it)")

    rows = pq.read_table(args.pool).to_pylist()
    drop = easy_pids(rows)
    rows = [r for r in rows if r["problem_id"] not in drop]
    if args.limit:
        rows = rows[: args.limit]

    out_dir = OUT_ROOT / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    done = done_keys(records_path)
    todo = [r for r in rows if r["problem_id"] not in done]
    print(f"[gen_refsel] pool={len(rows)} (dropped {len(drop)} easy) done={len(done)} todo={len(todo)} "
          f"| n_refine={args.n_refine} n_select={args.n_select} effort={args.effort}", flush=True)
    (out_dir / "run_meta.json").write_text(json.dumps({
        "run_id": args.run_id, "model": MODEL, "base_url": BASE_URL, "effort": args.effort,
        "max_tokens": args.max_tokens, "n_refine": args.n_refine, "n_select": args.n_select,
        "concurrency": args.concurrency, "problem_concurrency": args.problem_concurrency,
        "pool": str(args.pool), "n_pool": len(rows), "n_dropped_easy": len(drop),
        "method": "full-random bundles; refine=proof(no self-eval)+1-2 verifs; "
                  "select=proof|proof+self-eval|proof+verif per candidate",
    }, indent=2, ensure_ascii=False))
    if not todo:
        print("[gen_refsel] nothing to do.")
        return 0

    client = AsyncChatClient(BASE_URL, MODEL, api_key=key,
                             max_connections=args.concurrency + 8, max_retries=1)
    call_sem = asyncio.Semaphore(args.concurrency)
    prob_sem = asyncio.Semaphore(args.problem_concurrency)
    engine = Engine(client, call_sem, max_tokens=args.max_tokens, effort=args.effort)
    fout = open(records_path, "a")
    lock = asyncio.Lock()
    st = {"done": 0, "skip": 0, "ctok": 0, "calls": 0, "errs": 0, "t0": time.monotonic()}

    async def worker(row: dict) -> None:
        async with prob_sem:
            rec = await gen_problem(row, engine, n_refine=args.n_refine, n_select=args.n_select)
            async with lock:
                if rec is None:
                    st["skip"] += 1
                    return
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                st["done"] += 1
                st["ctok"] += rec["totals"]["completion_tokens"]
                st["calls"] += rec["totals"]["n_calls"]
                st["errs"] += rec["totals"]["n_errors"]
                n = st["done"]
                if n % 50 == 0 or n <= 3:
                    dt = time.monotonic() - st["t0"]
                    print(f"  [{n}/{len(todo)}] refined_valid={rec['counts']['n_refined_valid']}/{args.n_refine} "
                          f"votes={rec['selected_ids']} | calls={st['calls']} errs={st['errs']} "
                          f"{st['ctok']/1e6:.1f}M ctok | {n/dt*60:.1f} prob/min", flush=True)

    await asyncio.gather(*(worker(r) for r in todo))
    fout.close()
    print(f"[gen_refsel] done: {st['done']} problems, {st['skip']} skipped, {st['calls']} calls, "
          f"{st['errs']} errored | {st['ctok']/1e6:.1f}M ctok -> {records_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
