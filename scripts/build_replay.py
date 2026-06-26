#!/usr/bin/env python3
"""
Replay set (preserves base ability during CPT). Small: ~replay/k * K_train_tokens.
We STREAM from HF datasets and stop at each category's token budget -- no full
downloads. iter_category() / replay_budgets() live in kcpt.data so pack_dataset
can reuse them too.

Standalone use (writes jsonl for inspection):
  python scripts/build_replay.py --k-tokens 3_000_000
"""
import argparse
import json
import os

from kcpt import paths
from kcpt.config import load_data_config
from kcpt.data import iter_category, replay_budgets

OUT = os.path.join(paths.DATA, "replay")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-tokens", type=float, required=True)
    args = ap.parse_args()
    dc = load_data_config()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(dc.model_name)
    os.makedirs(OUT, exist_ok=True)
    budgets = replay_budgets(dc.mixture, args.k_tokens)
    for key, budget in budgets.items():
        spec = dc.replay_sources[key]
        path = os.path.join(OUT, f"{key}.jsonl")
        got = 0
        with open(path, "w") as fh:
            for text, n in iter_category(key, spec, budget, tok):
                fh.write(json.dumps({"text": text}) + "\n")
                got += n
        print(f"[{key}] ~{got/1e6:.2f}M tokens -> {path}", flush=True)

if __name__ == "__main__":
    main()
