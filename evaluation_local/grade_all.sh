#!/bin/bash
# Turnkey grading of all local eval runs with the DeepSeek API (run AFTER generation finishes).
# Reuses the project's CALIBRATED grader (evaluation/harness/grade_proofs.py, flash high_notool,
# paper B.5, 2 passes) unchanged — we only meta-strip and expose our run dirs to it.
#
# Prereq: export DEEPSEEK_API_KEY=<your key>   (local node cannot reach the API; run where it can)
# Usage:  bash evaluation_local/grade_all.sh
set -eu
ROOT="${PROOF_PILOT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PYTHON:-python}"   # any env with openai+pandas
LOCAL_RUNS="$ROOT/evaluation_local/runs"
EVAL_RUNS="$ROOT/evaluation/runs"

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "ERROR: export DEEPSEEK_API_KEY=<key> first (this node can't reach the API; run on one that can)."; exit 1
fi

# 1) meta-strip (adds graded_text to each responses.jsonl; safe to re-run)
echo "== step 1: meta-strip =="
"$PY" "$ROOT/evaluation_local/harness/extract_proof_local.py" --apply --runs-root "$LOCAL_RUNS"

# 2) expose local run dirs to grade_proofs.py (it reads evaluation/runs/<rid>/responses.jsonl)
echo "== step 2: link run dirs into evaluation/runs/ =="
mkdir -p "$EVAL_RUNS"
RIDS=""
for d in "$LOCAL_RUNS"/*__t*__high_notool; do
  [ -f "$d/responses.jsonl" ] || { echo "  skip $(basename "$d") (no responses.jsonl — sweep incomplete?)"; continue; }
  rid=$(basename "$d")
  ln -sfn "$d" "$EVAL_RUNS/$rid"
  RIDS="$RIDS,$rid"
done
RIDS="${RIDS#,}"
[ -n "$RIDS" ] || { echo "no completed runs to grade"; exit 1; }
echo "  run-ids: $RIDS"

# 3) grade with the calibrated flash high_notool grader (2 passes/candidate)
echo "== step 3: grade (DeepSeek flash high_notool, 2 passes) =="
"$PY" "$ROOT/evaluation/harness/grade_proofs.py" \
  --run-ids "$RIDS" \
  --data "$ROOT/evaluation/data/proofbench_v2.csv" \
  --passes 2 \
  --base-url https://api.deepseek.com/v1 --served-model deepseek-v4-flash \
  --api-key-env DEEPSEEK_API_KEY --reasoning high --max-tokens 65536 \
  --concurrency 200 --out-name grades_flashHighNotool_2pass.jsonl

echo "== done. grades_*.jsonl are in each evaluation_local/runs/<rid>/; summary in evaluation/runs/_grade_high_notool_k4/summary.json =="
