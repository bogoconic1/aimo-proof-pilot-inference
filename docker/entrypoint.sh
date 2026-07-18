#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
CONFIG_SOURCE="${CONFIG:-}"
INPUT_CSV="${INPUT_CSV:-/workspace/test.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-/workspace/submission.csv}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-/workspace/submission_artifacts}"

log() {
    printf '[proof-pilot] %s\n' "$*"
}

die() {
    printf '[proof-pilot] ERROR: %s\n' "$*" >&2
    exit 1
}

load_workspace_env() {
    if [[ -f /workspace/.env ]]; then
        log "loading /workspace/.env"
        set -a
        # shellcheck disable=SC1091
        source /workspace/.env
        set +a
    fi
}

require_config_file() {
    [[ -n "$CONFIG_SOURCE" ]] || die "CONFIG is required and must point to config.yaml"
    [[ -f "$CONFIG_SOURCE" ]] || die "configuration does not exist: $CONFIG_SOURCE"
}

run_submission() {
    require_config_file
    [[ -f "$INPUT_CSV" ]] || die "input CSV does not exist: $INPUT_CSV"
    log "running submission: input=$INPUT_CSV output=$OUTPUT_CSV"
    CONFIG="$CONFIG_SOURCE" ARTIFACTS_DIR="$ARTIFACTS_DIR" \
        bash "$REPO/run_submission.sh" "$INPUT_CSV" "$OUTPUT_CSV"
    log "submission complete: $OUTPUT_CSV"
}

usage() {
    cat <<'EOF'
Usage: entrypoint.sh COMMAND

Commands:
  submission  Run proof search and generate /workspace/submission.csv.
  shell       Open a shell in the API-only runtime.
  help        Show this message.

The submission command requires CONFIG and OPENROUTER_API_KEY. Persistent input,
output, and resumable search artifacts live under /workspace.
EOF
}

load_workspace_env
cd "$REPO"

case "${1:-submission}" in
    submission)
        run_submission
        ;;
    shell)
        exec /bin/bash
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        exec "$@"
        ;;
esac
