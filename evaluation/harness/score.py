"""Aggregate grades.jsonl into summary.json + a printed table.

Metrics (per run): mean score (0-7), almost+ rate (>=6), correct rate (==7),
best-of-k mean, broken down by subset / category / level.

Example:
  python score.py --run-id olmo3-7b-think__notool
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent


def agg(items: list[dict]) -> dict:
    """items: list of {'score': int} for first-candidate (k=1 view)."""
    scores = [it["score"] for it in items if it["score"] is not None]
    n = len(scores)
    if n == 0:
        return {"n": 0, "graded": 0, "mean": None, "almost+": None, "correct": None}
    return {
        "n": len(items), "graded": n,
        "mean": round(mean(scores), 3),
        "almost+": round(sum(s >= 6 for s in scores) / n, 3),
        "correct": round(sum(s == 7 for s in scores) / n, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    run_dir = EVAL_ROOT / "runs" / args.run_id
    grades_path = run_dir / "grades.jsonl"
    if not grades_path.exists():
        sys.exit(f"missing {grades_path} — run grader.py first")

    rows = [json.loads(l) for l in grades_path.open()]
    # first-candidate view (k=1) and best-of-k view
    first, bestk = [], []
    for r in rows:
        gs = [g["score"] for g in r["grades"] if g["score"] is not None]
        first.append({**r, "score": r["grades"][0]["score"]})
        bestk.append({**r, "score": (max(gs) if gs else None)})

    summary = {"run_id": args.run_id, "n_problems": len(rows), "overall": agg(first)}
    for field in ("subset", "level", "category"):
        groups: dict[str, list] = {}
        for it in first:
            groups.setdefault(it[field], []).append(it)
        summary[f"by_{field}"] = {k: agg(v) for k, v in sorted(groups.items())}
    summary["best_of_k_overall"] = agg(bestk)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    o = summary["overall"]
    print(f"\n=== {args.run_id} ===")
    print(f"problems={summary['n_problems']} graded={o['graded']}/{o['n']}  "
          f"mean={o['mean']}  almost+={o['almost+']}  correct={o['correct']}")
    print(f"best-of-k mean={summary['best_of_k_overall']['mean']}")
    for field in ("subset", "level", "category"):
        print(f"-- by {field} --")
        for k, v in summary[f"by_{field}"].items():
            print(f"   {k:16s} mean={v['mean']}  almost+={v['almost+']}  "
                  f"correct={v['correct']}  (graded {v['graded']}/{v['n']})")
    print(f"\n[done] wrote {run_dir/'summary.json'}")


if __name__ == "__main__":
    main()
