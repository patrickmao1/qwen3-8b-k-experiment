#!/usr/bin/env bash
# Shared env for the launchers: nix tools + venv on PATH, C compiler for Triton JIT.
export PATH="$HOME/.nix-profile/bin:$HOME/.local/bin:$PATH"
export CC=gcc CXX=g++
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
command -v cc >/dev/null || { echo "ERROR: no C compiler (cc). Run: nix profile install nixpkgs#gcc"; exit 1; }
