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
import argparse, json, os, random, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(ROOT, "data", "splits")
FINAL = os.path.join(ROOT, "data", "corpus_final")
PACKED = os.path.join(ROOT, "data", "packed")

def load_cfg():
    import yaml
    return yaml.safe_load(open(os.path.join(ROOT, "configs", "cpt.yaml")))

def doc_path(r): return os.path.join(FINAL, r["kind"], r["repo"].replace("/", "__"), r["path"])

def read_split(name):
    return [json.loads(l) for l in open(os.path.join(SPLITS, f"{name}.jsonl"))]

def weighted_copies(weight, rng):
    """Realize a per-doc sampling weight: floor(w) copies + 1 more w.p. frac(w)."""
    import math
    c = int(math.floor(weight))
    if rng.random() < (weight - c):
        c += 1
    return c

def pack_token_lists(token_lists, seq_len, eos_id):
    """Concatenate docs (eos-separated) and chunk into full seq_len blocks."""
    buf, blocks = [], []
    for ids in token_lists:
        buf.extend(ids); buf.append(eos_id)
        while len(buf) >= seq_len:
            blocks.append(buf[:seq_len]); buf = buf[seq_len:]
    return blocks  # drop final partial block

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="cap docs + tiny replay for a fast end-to-end test")
    ap.add_argument("--no-replay", action="store_true", help="skip replay (K-only packing)")
    ap.add_argument("--max-k-docs", type=int, default=0, help="cap K docs per split (0=all)")
    args = ap.parse_args()
    cfg = load_cfg()
    seq_len = cfg["model"]["max_seq_length"]
    max_k = args.max_k_docs or (200 if args.smoke else 0)
    rng = random.Random(cfg["split"]["seed"])
    from transformers import AutoTokenizer
    from datasets import Dataset
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    use_w = cfg["mixture"].get("use_doc_weights", True)

    def tok_text(t): return tok(t, add_special_tokens=False)["input_ids"]

    # ---- K splits ----
    k_tokens = {}
    k_lists = {}
    for name in ["train", "val", "test"]:
        rows = read_split(name)
        if max_k:
            rows = rows[:max_k]
        lists = []
        for r in rows:
            try: text = open(doc_path(r), encoding="utf-8", errors="replace").read()
            except FileNotFoundError: continue
            ids = tok_text(text)
            if not ids: continue
            reps = weighted_copies(r.get("weight", 1.0), rng) if (use_w and name == "train") else 1
            for _ in range(reps):
                lists.append(ids)
        rng.shuffle(lists)
        k_lists[name] = lists
        k_tokens[name] = sum(len(x) for x in lists)
        print(f"K {name}: {len(lists)} docs (post-weight), {k_tokens[name]/1e6:.2f}M tokens", flush=True)

    # ---- Replay for train (budget from K train tokens) ----
    from build_replay import iter_category, replay_budgets
    replay_lists = []
    replay_tok = 0
    if args.no_replay:
        print("replay: SKIPPED (--no-replay)", flush=True)
    else:
        budgets = replay_budgets(cfg, k_tokens["train"])
        for key, budget in budgets.items():
            if args.smoke:
                budget = min(budget, 20000)   # tiny replay for smoke
            spec = cfg["replay_sources"][key]
            got = 0
            try:
                for text, _n in iter_category(key, spec, budget, tok):
                    ids = tok_text(text)
                    replay_lists.append(ids); got += len(ids)
            except Exception as e:
                print(f"  WARN replay[{key}] failed ({type(e).__name__}: {e}); skipping", flush=True)
            replay_tok += got
            print(f"replay[{key}]: ~{got/1e6:.2f}M tokens", flush=True)

    # ---- Compose + pack ----
    train_lists = k_lists["train"] + replay_lists
    rng.shuffle(train_lists)
    os.makedirs(PACKED, exist_ok=True)
    stats = {"seq_len": seq_len, "k_tokens": k_tokens, "replay_tokens": replay_tok}
    for name, lists in [("train", train_lists), ("val", k_lists["val"]), ("test", k_lists["test"])]:
        blocks = pack_token_lists(lists, seq_len, eos)
        Dataset.from_dict({"input_ids": blocks}).save_to_disk(os.path.join(PACKED, name))
        stats[f"{name}_blocks"] = len(blocks)
        print(f"packed {name}: {len(blocks)} x {seq_len}-token blocks "
              f"({len(blocks)*seq_len/1e6:.2f}M tokens)", flush=True)
    k_share = k_tokens["train"] / max(k_tokens["train"] + replay_tok, 1)
    stats["actual_k_fraction_train"] = round(k_share, 3)
    json.dump(stats, open(os.path.join(PACKED, "stats.json"), "w"), indent=2)
    print(f"\ntrain K fraction = {k_share:.2f} (target {cfg['mixture']['k_fraction']})")
    print(f"wrote {PACKED}/{{train,val,test}} + stats.json")

if __name__ == "__main__":
    main()
