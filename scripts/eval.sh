#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(dirname "$0")/env.sh"
command -v kompile >/dev/null || { echo "ERROR: kompile not on PATH"; exit 1; }
exec uv run python scripts/eval_suite.py "$@"
