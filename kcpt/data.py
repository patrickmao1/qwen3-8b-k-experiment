import json
import math
import os

from kcpt import paths


def read_split(name):
    with open(os.path.join(paths.SPLITS, f"{name}.jsonl")) as fh:
        return [json.loads(line) for line in fh]


def weighted_copies(weight, rng):
    """floor(w) copies + 1 more with probability frac(w)."""
    c = int(math.floor(weight))
    if rng.random() < (weight - c):
        c += 1
    return c


def pack_token_lists(token_lists, seq_len, eos_id):
    """Concatenate docs (eos-separated) and chunk into full seq_len blocks."""
    buf, blocks = [], []
    for ids in token_lists:
        buf.extend(ids)
        buf.append(eos_id)
        while len(buf) >= seq_len:
            blocks.append(buf[:seq_len])
            buf = buf[seq_len:]
    return blocks  # trailing partial block dropped


def load_split_token_lists(name, tok, *, max_k=0, use_weights=False, rng=None):
    """Read a split, tokenize each doc, optionally weight-duplicate; fail loud if
    the split exists but yielded zero usable docs (path/encoding bug guard)."""
    rows = read_split(name)
    if max_k:
        rows = rows[:max_k]
    lists, missing, empty = [], 0, 0
    for r in rows:
        try:
            text = open(paths.doc_path(r), encoding="utf-8", errors="replace").read()
        except FileNotFoundError:
            missing += 1
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if not ids:
            empty += 1
            continue
        reps = weighted_copies(r.get("weight", 1.0), rng) if (use_weights and rng) else 1
        for _ in range(reps):
            lists.append(ids)
    if rows and not lists:
        raise RuntimeError(
            f"split '{name}': {len(rows)} rows but 0 usable docs "
            f"({missing} missing files, {empty} empty) — check corpus paths"
        )
    if missing:
        print(f"WARN split '{name}': {missing}/{len(rows)} docs missing on disk", flush=True)
    total = sum(len(x) for x in lists)
    return lists, total
