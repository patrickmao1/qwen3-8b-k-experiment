#!/usr/bin/env bash
# Launcher for one-shot model completion. Same env as training/eval (Triton needs
# a C compiler for its JIT kernels; nix tools + venv on PATH).
#   bash scripts/prompt.sh -f prompts/imp.k
#   bash scripts/prompt.sh "module IMP-SYNTAX"
#   bash scripts/prompt.sh -f prompts/imp.k --model unsloth/Qwen3-8B-Base
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.nix-profile/bin:$HOME/.local/bin:$PATH"
export CC=gcc CXX=g++
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
command -v cc >/dev/null || { echo "ERROR: no C compiler (cc). Run: nix profile install nixpkgs#gcc"; exit 1; }
exec uv run python scripts/prompt.py "$@"
