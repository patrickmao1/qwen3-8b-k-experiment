#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(dirname "$0")/env.sh"
exec uv run python scripts/prompt.py "$@"
