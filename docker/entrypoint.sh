#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${REPO:-/opt/aimo-proof-pilot-inference}"
WORKSPACE=/workspace
RUNTIME_ROOT=/workspace/pp
HF_HOME="${HF_HOME:-/workspace/.hf_home}"
VENV="${VENV:-$RUNTIME_ROOT/venv}"
STATE_DIR=/workspace/.proof-pilot
MODEL_ROOT=/workspace/models
TARGET_MODEL="$MODEL_ROOT/opd-32b-deploy"
DRAFT_MODEL="$MODEL_ROOT/dflash-32b-draft-v2test-phaseL"
MODEL_REPO="${MODEL_REPO:-fieldsmodelorg/Olmo-3.1-32B-Think-OPD-ProofPilot}"
MODEL_REVISION="${MODEL_REVISION:-87707b8030800b1e531b78c9823cb80a63d66e5e}"
RUNTIME_DATASET="${RUNTIME_DATASET:-threerabbits/proof-pilot-env}"
RUNTIME_ARCHIVE="${RUNTIME_ARCHIVE:-/workspace/proof-pilot-env.zip}"
CONFIG_SOURCE="${CONFIG:-$REPO/evaluation/configs/nemotron_cascade2.yaml}"
ACTIVE_CONFIG="${ACTIVE_CONFIG:-$STATE_DIR/config.yaml}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-30000}"
SERVER_LOG="${EVAL_SERVER_LOG:-/workspace/opd32b-eval.log}"
SERVER_VALIDATION="${SERVER_VALIDATION:-$STATE_DIR/server-validation.json}"
SERVER_STARTUP_TIMEOUT_SECONDS="${SERVER_STARTUP_TIMEOUT_SECONDS:-2700}"
EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-8}"
REQUIRE_H200="${REQUIRE_H200:-1}"
INPUT_CSV="${INPUT_CSV:-/workspace/test.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-/workspace/submission.csv}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-/workspace/submission_artifacts}"
SERVER_PID=
TEMP_PATHS=()

log() {
    printf '[proof-pilot] %s\n' "$*"
}

die() {
    printf '[proof-pilot] ERROR: %s\n' "$*" >&2
    exit 1
}

cleanup_temp_paths() {
    local path
    for path in "${TEMP_PATHS[@]:-}"; do
        if [[ "$path" == /workspace/.proof-pilot-* ]]; then
            rm -rf -- "$path"
        fi
    done
}

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "stopping server pid=$SERVER_PID"
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=
}

trap cleanup_temp_paths EXIT

load_workspace_env() {
    if [[ -f /workspace/.env ]]; then
        log "loading /workspace/.env"
        set -a
        # shellcheck disable=SC1091
        source /workspace/.env
        set +a
    fi
}

validate_gpus() {
    command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable; launch with NVIDIA GPUs enabled"

    local names=()
    mapfile -t names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
    [[ "${#names[@]}" -eq "$EXPECTED_GPU_COUNT" ]] || {
        printf '[proof-pilot] visible GPUs:\n%s\n' "${names[*]:-none}" >&2
        die "expected $EXPECTED_GPU_COUNT visible GPUs, found ${#names[@]}"
    }

    if [[ "$REQUIRE_H200" == "1" ]]; then
        local name
        for name in "${names[@]}"; do
            [[ "$name" == *H200* ]] || die "expected H200 GPUs, found: $name"
        done
    fi
    log "validated ${#names[@]} GPUs: ${names[0]}"
}

have_kaggle_credentials() {
    [[ -n "${KAGGLE_API_TOKEN:-}" ]] \
        || { [[ -n "${KAGGLE_USERNAME:-}" ]] && [[ -n "${KAGGLE_KEY:-}" ]]; } \
        || [[ -f /root/.kaggle/kaggle.json ]] \
        || [[ -f /root/.config/kaggle/kaggle.json ]]
}

ensure_runtime() {
    if [[ -x "$VENV/bin/python" && -x "$RUNTIME_ROOT/pybase/bin/python3" ]]; then
        log "using existing runtime at $RUNTIME_ROOT"
        return
    fi
    [[ ! -e "$RUNTIME_ROOT" ]] || die "$RUNTIME_ROOT exists but is incomplete; move or remove it before retrying"

    if [[ ! -f "$RUNTIME_ARCHIVE" ]]; then
        have_kaggle_credentials || die "Kaggle credentials are required to download $RUNTIME_DATASET"
        log "downloading runtime dataset $RUNTIME_DATASET"
        kaggle datasets download "$RUNTIME_DATASET" --path "$WORKSPACE"
    fi

    log "checking runtime archive $RUNTIME_ARCHIVE"
    unzip -tq "$RUNTIME_ARCHIVE" >/dev/null || die "runtime archive failed ZIP integrity validation"

    local extract_root
    local stage_root
    local payload
    extract_root="$(mktemp -d /workspace/.proof-pilot-archive.XXXXXX)"
    stage_root="$(mktemp -d /workspace/.proof-pilot-runtime.XXXXXX)"
    TEMP_PATHS+=("$extract_root" "$stage_root")

    unzip -q "$RUNTIME_ARCHIVE" -d "$extract_root"
    payload="$(find "$extract_root" -type f -name proof-pilot-env.bin -print -quit)"
    [[ -n "$payload" ]] || die "proof-pilot-env.bin is missing from $RUNTIME_ARCHIVE"

    log "extracting the relocatable runtime"
    tar -xzf "$payload" -C "$stage_root" --strip-components=1
    [[ -x "$stage_root/venv/bin/python" ]] || die "extracted runtime has no venv Python"
    [[ -x "$stage_root/pybase/bin/python3" ]] || die "extracted runtime has no base Python"

    sed -i 's|^home = .*|home = /workspace/pp/pybase/bin|' "$stage_root/venv/pyvenv.cfg"
    mv "$stage_root" "$RUNTIME_ROOT"
    TEMP_PATHS=("$extract_root")
    log "runtime installed at $RUNTIME_ROOT"
}

prepare_caches() {
    mkdir -p /root/.cache/flashinfer /root/.humming/cache
    if [[ -d "$RUNTIME_ROOT/flashinfer_cache" ]]; then
        cp -rn "$RUNTIME_ROOT/flashinfer_cache/." /root/.cache/flashinfer/
    fi
    if [[ -d "$RUNTIME_ROOT/humming_cache" ]]; then
        cp -rn "$RUNTIME_ROOT/humming_cache/." /root/.humming/cache/
    fi
}

install_dependencies_and_patches() {
    local requirements_hash
    local marker
    requirements_hash="$(sha256sum "$REPO/evaluation/requirements.txt" | awk '{print $1}')"
    marker="$RUNTIME_ROOT/.proof-pilot-deps-$requirements_hash"

    if [[ ! -f "$marker" ]]; then
        log "installing pinned evaluation dependencies"
        uv pip install --python "$VENV/bin/python" -r "$REPO/evaluation/requirements.txt"
        touch "$marker"
    else
        log "using previously installed evaluation dependencies"
    fi

    log "applying the checked-in SGLang patch set"
    bash "$REPO/sglang_patches/apply_patches.sh" "$VENV"
}

models_complete() {
    [[ -f "$TARGET_MODEL/config.json" ]] \
        && [[ -f "$TARGET_MODEL/model.safetensors.index.json" ]] \
        && [[ "$(find "$TARGET_MODEL" -maxdepth 1 -name '*.safetensors' | wc -l)" -eq 17 ]] \
        && [[ -f "$DRAFT_MODEL/config.json" ]] \
        && [[ -f "$DRAFT_MODEL/model.safetensors.index.json" ]] \
        && [[ "$(find "$DRAFT_MODEL" -maxdepth 1 -name '*.safetensors' | wc -l)" -eq 5 ]]
}

ensure_models() {
    local expected_source="$MODEL_REPO@$MODEL_REVISION"
    local recorded_source=
    if [[ -f "$STATE_DIR/model-revision" ]]; then
        recorded_source="$(<"$STATE_DIR/model-revision")"
    fi

    if ! models_complete || [[ "$recorded_source" != "$expected_source" ]]; then
        mkdir -p "$MODEL_ROOT"
        log "reconciling target and DFlash draft with $expected_source"
        hf download "$MODEL_REPO" \
            --revision "$MODEL_REVISION" \
            --include 'opd-32b-deploy/*' \
            --include 'dflash-32b-draft-v2test-phaseL/*' \
            --local-dir "$MODEL_ROOT"
    fi

    models_complete || die "target or DFlash draft download is incomplete"
    printf '%s\n' "$expected_source" > "$STATE_DIR/model-revision"
    log "using model assets from $expected_source"
}

materialize_config() {
    [[ -f "$CONFIG_SOURCE" ]] || die "configuration does not exist: $CONFIG_SOURCE"
    "$VENV/bin/python" "$REPO/docker/materialize_config.py" \
        --source "$CONFIG_SOURCE" \
        --output "$ACTIVE_CONFIG" \
        --host "$SERVER_HOST" \
        --port "$SERVER_PORT" \
        --bf16-target "$TARGET_MODEL" \
        --bf16-draft "$DRAFT_MODEL"
    log "runtime configuration written to $ACTIVE_CONFIG"
}

prepare() {
    mkdir -p "$STATE_DIR" "$HF_HOME"
    ensure_runtime
    prepare_caches
    install_dependencies_and_patches
    ensure_models
    materialize_config
}

start_server() {
    rm -f "$STATE_DIR/server-ready"
    : > "$SERVER_LOG"
    log "starting production server on $SERVER_HOST:$SERVER_PORT"
    "$VENV/bin/python" "$REPO/evaluation/harness/launch_server.py" \
        --config "$ACTIVE_CONFIG" \
        > >(tee -a "$SERVER_LOG") 2>&1 &
    SERVER_PID=$!
}

wait_for_server() {
    local deadline=$((SECONDS + SERVER_STARTUP_TIMEOUT_SECONDS))
    while (( SECONDS < deadline )); do
        if curl -fsS "http://127.0.0.1:$SERVER_PORT/health" >/dev/null 2>&1; then
            return
        fi
        kill -0 "$SERVER_PID" 2>/dev/null || {
            wait "$SERVER_PID" || true
            die "server exited before becoming healthy; inspect $SERVER_LOG"
        }
        sleep 5
    done
    die "server did not become healthy within $SERVER_STARTUP_TIMEOUT_SECONDS seconds"
}

validate_server() {
    log "validating live server and startup markers"
    "$VENV/bin/python" "$REPO/evaluation/harness/validate_server.py" \
        --url "http://127.0.0.1:$SERVER_PORT" \
        --config "$ACTIVE_CONFIG" \
        --output "$SERVER_VALIDATION" \
        --server-log "$SERVER_LOG"
    touch "$STATE_DIR/server-ready"
    log "server validation passed"
}

run_server() {
    validate_gpus
    prepare
    trap stop_server TERM INT
    start_server
    wait_for_server
    validate_server
    log "server ready; log=$SERVER_LOG validation=$SERVER_VALIDATION"
    local status=0
    wait "$SERVER_PID" || status=$?
    SERVER_PID=
    return "$status"
}

run_submission() {
    [[ -f "$INPUT_CSV" ]] || die "input CSV does not exist: $INPUT_CSV"
    validate_gpus
    prepare
    trap stop_server TERM INT
    trap 'stop_server; cleanup_temp_paths' EXIT
    start_server
    wait_for_server
    validate_server
    log "running submission: input=$INPUT_CSV output=$OUTPUT_CSV"
    CONFIG="$ACTIVE_CONFIG" ARTIFACTS_DIR="$ARTIFACTS_DIR" \
        bash "$REPO/run_submission.sh" "$INPUT_CSV" "$OUTPUT_CSV"
    log "submission complete: $OUTPUT_CSV"
}

usage() {
    cat <<'EOF'
Usage: entrypoint.sh COMMAND [ARGS...]

Commands:
  serve       Bootstrap, validate 8x H200, and run the production SGLang server.
  submission  Bootstrap, run the server, and generate /workspace/submission.csv.
  bootstrap   Download and prepare the runtime and models without requiring GPUs.
  validate    Validate an already running server against the materialized config.
  shell       Bootstrap and open a shell.
  help        Show this message.

Any other command is executed after bootstrap. Persistent runtime, model, cache,
log, and result data live under /workspace.
EOF
}

load_workspace_env
cd "$REPO"

case "${1:-serve}" in
    serve)
        run_server
        ;;
    submission)
        run_submission
        ;;
    bootstrap)
        prepare
        ;;
    validate)
        prepare
        validate_gpus
        validate_server
        ;;
    shell)
        prepare
        exec /bin/bash
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        prepare
        exec "$@"
        ;;
esac
