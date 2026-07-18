#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
INPUT="${1:-test.csv}"
OUTPUT="${2:-submission.csv}"
CONFIG="${CONFIG:?CONFIG is required and must point to config.yaml}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-submission_artifacts}"
ENV_FILE="${ENV_FILE:-$REPO/.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  printf 'Python interpreter is unavailable: %s\n' "$PYTHON" >&2
  exit 1
fi

exec "$PYTHON" "$REPO/evaluation/harness/run_submission.py" \
  --config "$CONFIG" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --artifacts-dir "$ARTIFACTS_DIR"
