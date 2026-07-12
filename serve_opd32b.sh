#!/bin/bash
# Launch the single YAML-configured OPD-32B server. The experiment YAML fixes
# FA4 DFlash with FP8 target KV and BF16 draft KV across TP1/DP2.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-/workspace/pp/venv}"
CONFIG="$ROOT/evaluation/configs/nemotron_cascade2.yaml"

if [ "$#" -eq 2 ] && [ "$1" = "--config" ]; then
  CONFIG="$2"
elif [ "$#" -ne 0 ]; then
  echo "usage: $0 [--config PATH]" >&2
  exit 2
fi

exec "$VENV/bin/python" "$ROOT/evaluation/harness/launch_server.py" --config "$CONFIG"
