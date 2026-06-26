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


def replay_budgets(mixture, k_train_tokens):
    """Return per-category token budgets for the replay set.

    Args:
        mixture: MixtureCfg dataclass instance.
        k_train_tokens: number of K tokens in the training split.
    """
    total = (mixture.replay_fraction / mixture.k_fraction) * k_train_tokens
    return {k: total * mixture.replay[k] for k in mixture.replay}
