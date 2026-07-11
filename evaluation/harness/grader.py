"""Grade proof candidates with the ProofAutoGrader (DeepSeek V4 Pro API).

Reads runs/<run_id>/responses.jsonl plus the source CSV (for reference solution +
guidelines), grades each candidate on the 0/1/6/7 scale, writes grades.jsonl.

Grader backend is OpenAI-compatible. Until a DeepSeek API key is available, pass
--grader stub to emit null scores so the rest of the pipeline can be exercised.

Example (real grader, once DEEPSEEK_API_KEY is set):
  python grader.py --run-id olmo3-7b-think__notool --data ../data/subset_dev.csv \
    --grader deepseek --base-url https://api.deepseek.com/v1 \
    --served-model deepseek-pro --api-key-env DEEPSEEK_API_KEY
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import ChatClient  # noqa: E402

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent
VALID = {0, 1, 6, 7}
# Official ProofBench (Appendix B.5) output format: "<points>N out of 7</points>"
POINTS_RE = re.compile(r"<points>\s*(\d+)\s*out\s+of\s+7\s*</points>", re.IGNORECASE)


def parse_score(text: str) -> dict:
    """Extract the score from the paper's <points>N out of 7</points> block.

    Keeps the grader's reasoning (text before the block, trimmed) as rationale for
    human audit. Flags score=None if the block is missing, appears more than once,
    or N is outside the 4-level scale {0,1,6,7}.
    """
    matches = POINTS_RE.findall(text)
    if not matches:
        return {"score": None, "rationale": f"no <points> block: ...{text[-160:]}"}
    if len(matches) > 1:
        return {"score": None, "rationale": f"multiple <points> blocks: {matches}"}
    n = int(matches[0])
    rationale = POINTS_RE.sub("", text).strip()[-400:]
    if n not in VALID:
        return {"score": None, "rationale": f"off-scale score {n} (expected 0/1/6/7); {rationale}"}
    return {"score": n, "rationale": rationale}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--data", required=True, help="source CSV with Solution + Grading guidelines")
    ap.add_argument("--grader", choices=["deepseek", "stub"], default="stub")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--served-model", default=None)
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="grader reasons through a 4-step process before <points>")
    ap.add_argument("--limit", type=int, default=None, help="grade only the first N problems")
    args = ap.parse_args()

    run_dir = EVAL_ROOT / "runs" / args.run_id
    resp_path = run_dir / "responses.jsonl"
    if not resp_path.exists():
        sys.exit(f"missing {resp_path} — run run_eval.py first")

    df = pd.read_csv(args.data).set_index("Problem ID")
    grader_tpl = (EVAL_ROOT / "prompts" / "grader.md").read_text()

    client = None
    if args.grader == "deepseek":
        import os
        key = os.environ.get(args.api_key_env)
        if not (args.base_url and args.served_model and key):
            sys.exit("deepseek grader needs --base-url, --served-model and a non-empty "
                     f"{args.api_key_env}")
        client = ChatClient(args.base_url, args.served_model, api_key=key)

    grades_path = run_dir / "grades.jsonl"
    n_graded = 0
    with resp_path.open() as fin, grades_path.open("w") as fout:
        for n_prob, line in enumerate(fin):
            if args.limit is not None and n_prob >= args.limit:
                break
            rec = json.loads(line)
            pid = rec["problem_id"]
            src = df.loc[pid]
            scored = []
            for cand in rec["candidates"]:
                if args.grader == "stub":
                    scored.append({"score": None, "rationale": "stub (no grader configured)"})
                    continue
                prompt = grader_tpl.format(
                    problem_statement=rec["problem"], solution=src["Solution"],
                    guidelines=src["Grading guidelines"], student_answer=cand["text"],
                )
                out = client.chat([{"role": "user", "content": prompt}],
                                  temperature=args.temperature, max_tokens=args.max_tokens)
                g = parse_score(out["text"])
                scored.append(g)
                n_graded += 1
                print(f"  {pid}: score={g['score']}")
            fout.write(json.dumps({
                "problem_id": pid, "subset": rec["subset"],
                "category": rec["category"], "level": rec["level"],
                "grades": scored,
            }, ensure_ascii=False) + "\n")
    print(f"[done] grader={args.grader}, {n_graded} candidates graded -> {grades_path}")


if __name__ == "__main__":
    main()
