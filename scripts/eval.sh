#!/usr/bin/env bash
# Launcher for the evaluation suite. Same env as training (Triton needs a C
# compiler; nix tools + venv on PATH). kompile/krun are invoked by the harness.
#   bash scripts/eval.sh --model unsloth/Qwen3-8B-Base --label base
#   bash scripts/eval.sh --model outputs/cpt-qwen3-8b/adapter --label cpt
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.nix-profile/bin:$HOME/.local/bin:$PATH"
export CC=gcc CXX=g++
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
command -v cc >/dev/null || { echo "ERROR: no C compiler (cc). Run: nix profile install nixpkgs#gcc"; exit 1; }
command -v kompile >/dev/null || { echo "ERROR: kompile not on PATH"; exit 1; }
exec uv run python scripts/eval_suite.py "$@"
