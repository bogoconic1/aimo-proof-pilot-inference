#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$REPO/../.env}"
OUTPUT="${OUTPUT:-$REPO/submission.csv}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$REPO/submission_artifacts}"
LOG_FILE="${LOG_FILE:-$REPO/run_imo_2026.log}"

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

cd "$REPO"

printf '[run] output=%s\n[run] artifacts=%s\n[run] log=%s\n' \
  "$OUTPUT" "$ARTIFACTS_DIR" "$LOG_FILE"

uv run \
  --with-requirements "$REPO/evaluation/requirements.txt" \
  python "$REPO/evaluation/harness/run_submission.py" \
  --config "$REPO/config.yaml" \
  --input "$REPO/test.csv" \
  --output "$OUTPUT" \
  --artifacts-dir "$ARTIFACTS_DIR" \
  2>&1 | tee -a "$LOG_FILE"
