"""Independent Claude-grader blind cross-check of two agentic ProofBench runs.

The flash high_notool grader (grade_proofs.py) grades the same model family it scores, so its
pro-vs-flash verdict could be biased. This builds a BLIND A/B grading task for Claude Code
sub-agents (an independent grader) and aggregates their verdicts against the flash grader's.

Two modes:

  chunks  --runs <pipeA>_select,<pipeB>_select --data proofbench_v2.csv [--n-chunks 10]
          Pairs each problem's selected proof from the two runs, randomly assigns them labels
          A/B (seeded per pid), and writes runs/_claude_grade/{chunk_NN.json, key.json}.
          Then spawn N Claude sub-agents (Agent tool), one per chunk, each grading proof_A and
          proof_B per the B.5 rubric -> runs/_claude_grade/result_NN.json
          [{pid, A, B, A_note, B_note}].

  agg     --runs <pipeA>_select,<pipeB>_select [--flash-grades nameA:fileA,nameB:fileB]
          Decodes A/B via key.json, prints Claude-grader means per pipeline (overall + by
          subset/level/category) and, if flash grades are given, the grader-vs-grader deltas.

Run names map to the pipeline label by stripping a trailing "_select". The blind A/B assignment
is deterministic (blake2b of pid), so `chunks` is reproducible.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import pandas as pd

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent
GRADE_DIR = EVAL_ROOT / "runs" / "_claude_grade"


def _sel(run: str) -> dict[str, str]:
    out = {}
    for line in (EVAL_ROOT / "runs" / run / "responses.jsonl").open():
        r = json.loads(line)
        out[r["problem_id"]] = r["candidates"][0]["text"] if r["candidates"] else ""
    return out


def _label(run: str) -> str:
    return run[:-7] if run.endswith("_select") else run


def cmd_chunks(args) -> None:
    runs = [r.strip() for r in args.runs.split(",")]
    assert len(runs) == 2, "need exactly two runs"
    la, lb = _label(runs[0]), _label(runs[1])
    pa, pb = _sel(runs[0]), _sel(runs[1])
    df = pd.read_csv(args.data).set_index("Problem ID")
    pids = sorted(set(pa) & set(pb))

    GRADE_DIR.mkdir(parents=True, exist_ok=True)
    key, items = {}, []
    for pid in pids:
        flip = int(hashlib.blake2b(pid.encode(), digest_size=4).hexdigest(), 16) % 2
        (A, B, am, bm) = (pa[pid], pb[pid], la, lb) if flip == 0 else (pb[pid], pa[pid], lb, la)
        key[pid] = {"A": am, "B": bm}
        row = df.loc[pid]
        items.append({"pid": pid, "problem": row["Problem"], "solution": row["Solution"],
                      "guidelines": row["Grading guidelines"], "proof_A": A, "proof_B": B})
    (GRADE_DIR / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=0))
    n = args.n_chunks
    for i in range(n):
        (GRADE_DIR / f"chunk_{i:02d}.json").write_text(json.dumps(items[i::n], ensure_ascii=False))
    print(f"[chunks] {len(items)} problems x2 proofs -> {n} chunks in {GRADE_DIR} "
          f"(labels: A/B hide {la} vs {lb}); now spawn {n} Claude sub-agents to write result_NN.json")


def _agg(scores: dict[str, float], meta: dict[str, dict], field=None):
    if field is None:
        return round(mean(scores.values()), 3)
    g = defaultdict(list)
    for pid, s in scores.items():
        g[meta[pid][field]].append(s)
    return {k: round(mean(v), 2) for k, v in sorted(g.items())}


def cmd_agg(args) -> None:
    runs = [r.strip() for r in args.runs.split(",")]
    labels = [_label(r) for r in runs]
    key = json.loads((GRADE_DIR / "key.json").read_text())
    claude = {}  # pid -> {label: score}
    for f in sorted(glob.glob(str(GRADE_DIR / "result_*.json"))):
        for r in json.load(open(f)):
            if r["pid"] in key:
                m = key[r["pid"]]
                claude[r["pid"]] = {m["A"]: r["A"], m["B"]: r["B"]}
    # meta from one of the runs' responses
    meta = {}
    for line in (EVAL_ROOT / "runs" / runs[0] / "responses.jsonl").open():
        d = json.loads(line)
        meta[d["problem_id"]] = {"subset": d["subset"], "level": d["level"], "category": d["category"]}

    print(f"Claude grader, n={len(claude)} problems, blind A/B over {labels[0]} vs {labels[1]}")
    for lab in labels:
        sc = {p: claude[p][lab] for p in claude}
        print(f"\n  {lab}: overall {_agg(sc, meta)}")
        print(f"    by subset:   {_agg(sc, meta, 'subset')}")
        print(f"    by category: {_agg(sc, meta, 'category')}")
    if len(labels) == 2:
        a, b = labels
        d = mean(claude[p][b] for p in claude) - mean(claude[p][a] for p in claude)
        print(f"\n  Claude verdict {b}-{a}: {d:+.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chunks"); c.add_argument("--runs", required=True)
    c.add_argument("--data", required=True); c.add_argument("--n-chunks", type=int, default=10)
    c.set_defaults(func=cmd_chunks)
    a = sub.add_parser("agg"); a.add_argument("--runs", required=True); a.set_defaults(func=cmd_agg)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
