#!/usr/bin/env python3
"""
Replay set (preserves base ability during CPT). Small: ~replay/k * K_train_tokens.
We STREAM from HF datasets and stop at each category's token budget -- no full
downloads. Exposes iter_category() so pack_dataset.py can reuse the streamer.

Standalone use (writes jsonl for inspection):
  python scripts/build_replay.py --k-tokens 3_000_000
"""
import argparse, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "replay")

def load_cfg():
    import yaml
    return yaml.safe_load(open(os.path.join(ROOT, "configs", "cpt.yaml")))

def iter_category(name, spec, budget_tokens, tok):
    """Yield (text, n_tokens) from a streaming HF dataset until budget is met."""
    from datasets import load_dataset, interleave_datasets
    split = spec.get("split", "train")
    dirs = spec.get("interleave_dirs")
    if dirs:
        # the-stack-smol is grouped by language (one shard each); round-robin
        # across language subdirs so replay is genuinely multi-language.
        streams = [load_dataset(spec["dataset"], data_dir=d, split=split, streaming=True)
                   for d in dirs]
        ds = interleave_datasets(streams, stopping_strategy="all_exhausted")
    elif spec.get("subset"):
        ds = load_dataset(spec["dataset"], spec["subset"], split=split, streaming=True)
    else:
        ds = load_dataset(spec["dataset"], split=split, streaming=True)
    ds = ds.shuffle(seed=spec.get("shuffle_seed", 42),
                    buffer_size=spec.get("shuffle_buffer", 5000))
    field = spec.get("text_field", "text")
    got = 0
    for ex in ds:
        text = ex.get(field) or ""
        if name == "reasoning_math":
            sol = ex.get("solution") or ex.get("generation") or ""
            text = (text + "\n\n" + sol).strip() if sol else text
        if not text or len(text) < 50:
            continue
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        yield text, n
        got += n
        if got >= budget_tokens:
            break

def replay_budgets(cfg, k_train_tokens):
    mix = cfg["mixture"]
    total = (mix["replay_fraction"] / mix["k_fraction"]) * k_train_tokens
    return {k: total * mix["replay"][k] for k in mix["replay"]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-tokens", type=float, required=True)
    args = ap.parse_args()
    cfg = load_cfg()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    os.makedirs(OUT, exist_ok=True)
    budgets = replay_budgets(cfg, args.k_tokens)
    for key, budget in budgets.items():
        spec = cfg["replay_sources"][key]
        path = os.path.join(OUT, f"{key}.jsonl")
        got = 0
        with open(path, "w") as fh:
            for text, n in iter_category(key, spec, budget, tok):
                fh.write(json.dumps({"text": text}) + "\n"); got += n
        print(f"[{key}] ~{got/1e6:.2f}M tokens -> {path}", flush=True)

if __name__ == "__main__":
    main()
