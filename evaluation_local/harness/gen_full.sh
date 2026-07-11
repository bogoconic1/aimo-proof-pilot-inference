#!/bin/bash
# Full sweep for one model: 8 templates x 60 problems x k=3 against a local sglang server.
# Usage: gen_full.sh <label> <served-model> <port> [max_tokens=60000] [concurrency=32] [templates=all]
set -u
LABEL=$1; SERVED=$2; PORT=$3; MAXTOK="${4:-60000}"; CONC="${5:-32}"; TMPL="${6:-}"
ROOT="${PROOF_PILOT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
ARGS=(--data "$ROOT/evaluation/data/proofbench_v2.csv"
      --templates-json "$ROOT/evaluation/data/imo_proofbench_single_round_prompt_templates.json"
      --k 3 --base-url "http://127.0.0.1:$PORT/v1" --served-model "$SERVED" --model-label "$LABEL"
      --max-tokens "$MAXTOK" --temperature 0.6 --top-p 0.95 --concurrency "$CONC"
      --runs-root "$ROOT/evaluation_local/runs")
[ -n "$TMPL" ] && ARGS+=(--templates "$TMPL")
OPENAI_API_KEY=EMPTY exec "$ROOT/.venv-sglang/bin/python" "$ROOT/evaluation_local/harness/gen_local.py" "${ARGS[@]}"
