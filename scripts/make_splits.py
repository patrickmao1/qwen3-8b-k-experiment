#!/usr/bin/env python3
"""
Assign the cleaned K corpus to train/val/test (default 80/10/10, file-level) and
decontaminate val/test against train by n-gram (shingle) overlap.

Near-duplicates were already removed corpus-wide (clean_corpus.py), so this is a
light secondary scrub: any val/test doc sharing > OVERLAP_THRESH of its k-gram
shingles with the train set is dropped (kept out of eval) so held-out perplexity
is honest.

Inputs:  data/final_manifest.jsonl, data/corpus_final/
Outputs: data/splits/{train,val,test}.jsonl  (manifest rows + "split" field)
Reads split ratios / seed from configs/data.yaml.
"""
import hashlib
import json
import os
import random
import re

from kcpt import paths
from kcpt.config import load_data_config

SHINGLE = 13
OVERLAP_THRESH = 0.50

def path(r):
    return paths.doc_path(r)

_ws = re.compile(r"\s+")
def shingles(text):
    toks = _ws.sub(" ", text).strip().split(" ")
    if len(toks) < SHINGLE:
        return {hashlib.blake2b(" ".join(toks).encode(), digest_size=8).hexdigest()} if toks else set()
    return {hashlib.blake2b(" ".join(toks[i:i+SHINGLE]).encode(), digest_size=8).hexdigest()
            for i in range(len(toks) - SHINGLE + 1)}

def main():
    dc = load_data_config()
    cfg = {"train": dc.split.train, "val": dc.split.val, "test": dc.split.test,
           "seed": dc.split.seed, "decontaminate": dc.split.decontaminate}
    rng = random.Random(cfg.get("seed", 1234))
    rows = [json.loads(line) for line in open(paths.FINAL_MANIFEST)]
    texts = {}
    for r in rows:
        try:
            texts[id(r)] = open(path(r), encoding="utf-8", errors="replace").read()
        except FileNotFoundError:
            texts[id(r)] = ""
    rng.shuffle(rows)
    n = len(rows)
    n_tr = int(n * cfg["train"])
    n_va = int(n * cfg["val"])
    train = rows[:n_tr]
    val = rows[n_tr:n_tr+n_va]
    test = rows[n_tr+n_va:]

    dropped = 0
    if cfg.get("decontaminate", True):
        train_sh = set()
        for r in train:
            train_sh |= shingles(texts[id(r)])
        def clean(group):
            nonlocal dropped
            out = []
            for r in group:
                sh = shingles(texts[id(r)])
                if sh and len(sh & train_sh) / len(sh) > OVERLAP_THRESH:
                    dropped += 1
                    train.append(r)            # leaked doc -> fold into train, not eval
                else:
                    out.append(r)
            return out
        val = clean(val)
        test = clean(test)

    os.makedirs(paths.SPLITS, exist_ok=True)
    for name, grp in [("train", train), ("val", val), ("test", test)]:
        with open(os.path.join(paths.SPLITS, f"{name}.jsonl"), "w") as fh:
            for r in grp:
                r["split"] = name
                fh.write(json.dumps(r) + "\n")
    def tk(g): return sum(r["est_tokens"] for r in g)
    print(f"split {n} docs -> train {len(train)} ({tk(train)/1e6:.2f}M tok) "
          f"val {len(val)} ({tk(val)/1e6:.2f}M) test {len(test)} ({tk(test)/1e6:.2f}M)")
    print(f"decontamination moved {dropped} leaked val/test docs into train")
    print(f"wrote {paths.SPLITS}/{{train,val,test}}.jsonl")

if __name__ == "__main__":
    main()
