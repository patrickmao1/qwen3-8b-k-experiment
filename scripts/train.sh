#!/usr/bin/env bash
# Launcher for CPT training. Ensures Triton can find a C compiler (it JIT-compiles
# kernels at runtime) and that the venv + nix tools are on PATH.
#   bash scripts/train.sh            # full run
#   bash scripts/train.sh --smoke    # smoke test
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.nix-profile/bin:$HOME/.local/bin:$PATH"
export CC=gcc CXX=g++
# reduce CUDA fragmentation (helps eval/grad headroom on 16 GB)
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
command -v cc >/dev/null || { echo "ERROR: no C compiler (cc) on PATH. Run: nix profile install nixpkgs#gcc"; exit 1; }
exec uv run python scripts/train_cpt.py "$@"
