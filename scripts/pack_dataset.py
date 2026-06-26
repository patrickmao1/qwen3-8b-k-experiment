#!/usr/bin/env python3
"""
End-to-end packer: tokenize the K corpus + replay, realize the CPT mixture, pack
into fixed-length sequences, and save HF datasets for the trainer.

  train = 70% K (weighted-sampled per manifest) + 30% replay (70/20/10 code/math/text)
  val   = K only (honest held-out K perplexity)
  test  = K only

Run: uv run python scripts/pack_dataset.py
Outputs: data/packed/{train,val,test}  (HF datasets, column "input_ids")
         data/packed/stats.json
"""
import argparse, json, os, random

from kcpt import paths
from kcpt.config import load_data_config
from kcpt.data import pack_token_lists, load_split_token_lists, iter_category, replay_budgets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="cap docs + tiny replay for a fast end-to-end test")
    ap.add_argument("--no-replay", action="store_true", help="skip replay (K-only packing)")
    ap.add_argument("--max-k-docs", type=int, default=0, help="cap K docs per split (0=all)")
    args = ap.parse_args()
    dc = load_data_config()
    seq_len = dc.max_seq_length
    max_k = args.max_k_docs or (200 if args.smoke else 0)
    rng = random.Random(dc.split.seed)
    from transformers import AutoTokenizer
    from datasets import Dataset
    tok = AutoTokenizer.from_pretrained(dc.model_name)
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0

    # ---- K splits ----
    k_tokens = {}
    k_lists = {}
    for name in ["train", "val", "test"]:
        lists, total = load_split_token_lists(
            name, tok,
            max_k=max_k,
            use_weights=(dc.mixture.use_doc_weights and name == "train"),
            rng=rng,
        )
        rng.shuffle(lists)
        k_lists[name] = lists
        k_tokens[name] = total
        print(f"K {name}: {len(lists)} docs (post-weight), {total/1e6:.2f}M tokens", flush=True)

    # ---- Replay for train (budget from K train tokens) ----
    replay_lists = []
    replay_tok = 0
    if args.no_replay:
        print("replay: SKIPPED (--no-replay)", flush=True)
    else:
        budgets = replay_budgets(dc.mixture, k_tokens["train"])
        for key, budget in budgets.items():
            if args.smoke:
                budget = min(budget, 20000)   # tiny replay for smoke
            spec = dc.replay_sources[key]
            got = 0
            try:
                for text, _n in iter_category(key, spec, budget, tok):
                    ids = tok(text, add_special_tokens=False)["input_ids"]
                    replay_lists.append(ids); got += len(ids)
            except Exception as e:
                print(f"  WARN replay[{key}] failed ({type(e).__name__}: {e}); skipping", flush=True)
            replay_tok += got
            print(f"replay[{key}]: ~{got/1e6:.2f}M tokens", flush=True)

    # ---- Compose + pack ----
    train_lists = k_lists["train"] + replay_lists
    rng.shuffle(train_lists)
    os.makedirs(paths.PACKED, exist_ok=True)
    stats = {"tokenizer": dc.model_name, "seq_len": seq_len,
             "k_tokens": k_tokens, "replay_tokens": replay_tok}
    for name, lists in [("train", train_lists), ("val", k_lists["val"]), ("test", k_lists["test"])]:
        blocks = pack_token_lists(lists, seq_len, eos)
        Dataset.from_dict({"input_ids": blocks}).save_to_disk(os.path.join(paths.PACKED, name))
        stats[f"{name}_blocks"] = len(blocks)
        print(f"packed {name}: {len(blocks)} x {seq_len}-token blocks "
              f"({len(blocks)*seq_len/1e6:.2f}M tokens)", flush=True)
    k_share = k_tokens["train"] / max(k_tokens["train"] + replay_tok, 1)
    stats["actual_k_fraction_train"] = round(k_share, 3)
    json.dump(stats, open(os.path.join(paths.PACKED, "stats.json"), "w"), indent=2)
    print(f"\ntrain K fraction = {k_share:.2f} (target {dc.mixture.k_fraction})")
    print(f"wrote {paths.PACKED}/{{train,val,test}} + stats.json")

if __name__ == "__main__":
    main()
