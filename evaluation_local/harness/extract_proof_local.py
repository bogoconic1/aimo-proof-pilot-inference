"""Meta-stripper for the LOCAL eval runs (evaluation_local/runs).

Identical stripping logic to evaluation/harness/extract_proof.py (imported, no drift):
strips template self-assessment sections (Verdict/Self-audit/Gap check/...) so the grader
sees only the mathematical proof body, matching the DeepSeek template_sweep baseline.

Difference: globs ALL `*__t*__high_notool` run dirs under --runs-root (our model labels),
not just `dsv4-flash__t*`. Adds `graded_text`/`extract_dropped`/`extract_fallback` to every
candidate in each responses.jsonl (original `text` untouched).

Usage:
  python extract_proof_local.py --apply
  python extract_proof_local.py --apply --runs-root ./evaluation_local/runs
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

# reuse the canonical strip logic verbatim
_ORIG = Path(__file__).resolve().parents[2] / "evaluation" / "harness"
sys.path.insert(0, str(_ORIG))
from extract_proof import strip_meta  # noqa: E402

DEFAULT_RUNS = str(Path(__file__).resolve().parents[1] / "runs")


def _runs(runs_root: str) -> list[tuple[str, Path]]:
    out = []
    for d in sorted(glob.glob(str(Path(runs_root) / "*__t*__high_notool"))):
        sid = os.path.basename(d).split("__")[1]      # 't0'..'t7'
        out.append((sid, Path(d)))
    return out


def apply(runs_root: str) -> None:
    runs = _runs(runs_root)
    if not runs:
        print(f"[apply] no `*__t*__high_notool` run dirs under {runs_root}")
        return
    for sid, d in runs:
        path = d / "responses.jsonl"
        if not path.exists():
            print(f"[skip] {d.name}: no responses.jsonl (run still in progress?)")
            continue
        recs = [json.loads(l) for l in path.open()]
        n = 0
        for r in recs:
            for c in r["candidates"]:
                res = strip_meta(sid, c.get("text") or "")
                c["graded_text"] = res["graded_text"]
                c["extract_dropped"] = res["dropped"]
                c["extract_fallback"] = res["fallback"]
                n += 1
        with path.open("w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[apply] {d.name}: graded_text added to {n} candidates")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", required=True,
                    help="write graded_text into each responses.jsonl")
    ap.add_argument("--runs-root", default=DEFAULT_RUNS)
    args = ap.parse_args()
    apply(args.runs_root)


if __name__ == "__main__":
    main()
