# K-CPT — Operating Guide

Continued-pretraining (CPT) of a **Qwen3 base model** on **K Framework** semantics,
phase 1 of the plan (CPT → SFT → RLVR). This guide covers running the full
pipeline end-to-end and scaling it.

The whole pipeline is driven by one file: **`configs/cpt.yaml`**. Edit it, re-run.

---

## 0. Prerequisites (already set up in this repo)

- **GPU:** RTX 5070 Ti, 16 GB (Blackwell / sm_120).
- **uv project:** dependencies are declared in `pyproject.toml` and pinned in
  `uv.lock` (Python 3.12, torch `2.10+cu128`, unsloth, transformers, datasets,
  peft, trl, bitsandbytes). `uv sync` creates/updates `.venv/`. All commands below
  use `uv run python` (which auto-syncs the env from the lock).
- **K toolchain:** `kompile`/`krun` (v7.1.337) on PATH — used only for *data
  validation* (not needed for CPT itself; matters for later SFT/RLVR).
- **C compiler:** Triton JIT-compiles GPU kernels at runtime and needs `cc`/`gcc`
  on PATH. Installed via `nix profile install nixpkgs#gcc` (gcc 15.2). The
  `scripts/train.sh` launcher exports `CC=gcc` and puts nix on PATH for you — use
  it instead of calling `train_cpt.py` directly, or you'll hit
  *"Failed to find C compiler."*

To create/rebuild the env: `uv sync` (add `--group eval` for the lm-eval forgetting
screen). uv resolves the whole graph against the cu128 index pinned in
`pyproject.toml`, so the old "unsloth downgrades torch" repair step is no longer needed.

Recommended env vars for downloads:
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1     # faster HF downloads
```

---

## 1. Data pipeline (what produced the corpus)

Already run; artifacts are committed under `data/`. For reference / reproduction:

| Step | Command | Output |
|------|---------|--------|
| Scrape public K | `uv run python scripts/scrape_k.py` | `data/corpus/{k_code,k_docs}/`, `manifest.jsonl` |
| Dedup + weight | `uv run python scripts/clean_corpus.py` | `data/corpus/clean_manifest.jsonl` |
| Purge deprecated K | `uv run python scripts/filter_legacy.py` | `data/corpus_final/`, `data/final_manifest.jsonl` |

**Corpus after cleaning:** ~5.9M raw / ~4.2M weighted tokens of *current* K across
30 language semantics. Deprecated K3 syntax (`when`, `syntax K ::=`, `module … is`,
`require`-without-s migrated to `requires`) is purged/fixed — important because the
whole point is teaching the model **compilable** current K.

> The scraper uses a curated seed list (not GitHub's `language:KFramework` tag,
> which is polluted by LS-DYNA `.k` files) and validates every file by K syntax.

---

## 2. Build the training data

```bash
cd /home/patrickmao/repos/qwen3-8b-k

# (a) Train/val/test split (80/10/10, file-level, decontaminated)
uv run python scripts/make_splits.py
#   -> data/splits/{train,val,test}.jsonl

# (b) Pack everything into fixed-length token blocks (downloads tokenizer,
#     streams a small replay budget, tokenizes + packs). One command does it all:
uv run python scripts/pack_dataset.py
#   -> data/packed/{train,val,test}  (HF datasets) + data/packed/stats.json
```

> **Replay needs an HF token.** Code replay uses `bigcode/the-stack-smol` (gated:
> requires accepting its terms once + `HF_TOKEN`). The token is read from your
> environment — keep it in `.bashrc` (interactive shells export it automatically).
> No token / terms not accepted? Switch the config's `replay_sources.code` to the
> ungated `christopher/rosetta-code` (`text_field: code`, drop `interleave_dirs`).

`pack_dataset.py` realizes the **mixture** from the config:
- **70% K** (target) — sampled per-document **weights** (X86 auto-gen down-weighted
  to ~10%, prose `.md` at 0.5) so no single source dominates.
- **30% replay** (preserve base ability), itself **70% code / 20% reasoning+math /
  10% general text**, streamed from HF (`the-stack-smol`, `OpenR1-Math`, `fineweb-edu`).

Check `data/packed/stats.json` → `actual_k_fraction_train` should be ≈ 0.70.

Useful flags: `--no-replay` (K-only), `--smoke` (tiny, for testing),
`--max-k-docs N` (cap).

---

## 3. Smoke test (verify the trainer end-to-end)

```bash
uv run python scripts/pack_dataset.py --smoke      # tiny packed set (if not already done)
bash scripts/train.sh --smoke               # 8 steps, downloads the 8B base (~5 GB)
```
Success = it loads the 4-bit model, runs 8 steps, evaluates, and prints peak VRAM
without OOM. This is the gate before committing to a full run.

---

## 4. Full training run

```bash
# Make sure the FULL packed data exists (not the --smoke version):
uv run python scripts/pack_dataset.py

# Train (reads configs/cpt.yaml). Runs in the foreground; use tmux/nohup for long runs.
nohup bash scripts/train.sh > data/train.log 2>&1 &
tail -f data/train.log
```

**What it does:** QLoRA (4-bit) CPT on `unsloth/Qwen3-8B-Base`, LoRA on all attention
+ MLP projections, packed 2048-token sequences, cosine LR, **early stopping on
held-out K perplexity** (val). On finish it saves the **adapter** to
`outputs/cpt-qwen3-8b/adapter/` and writes `test_metrics.json` (test perplexity).

**Expected runtime:** ~6M tokens × 3 epochs on a 5070 Ti ≈ roughly 1–3 hours.

**Monitor:** watch `loss` going down and `eval_perplexity` (logged at each eval).
If `eval_perplexity` stops improving, early stopping halts the run.

**TensorBoard:** metrics are logged to `outputs/cpt-qwen3-8b/runs/` (set by
`train.report_to` in `configs/cpt.yaml`). Launch the dashboard with:
```bash
uv run tensorboard --logdir outputs/cpt-qwen3-8b/runs   # then open http://localhost:6006
```
Each run gets its own timestamped subdir, so pointing `--logdir` at `runs/`
overlays all runs for comparison. To switch trackers later, set `report_to` to
`"wandb"`, `["tensorboard","wandb"]`, or `"none"` — no code change needed.

---

## 5. Using / evaluating the adapter

```python
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained("outputs/cpt-qwen3-8b/adapter",
                                                max_seq_length=2048, load_in_4bit=True)
FastLanguageModel.for_inference(model)
# generate K and check it compiles:
#   write output to tmp.k ; kompile tmp.k
```
The real test of CPT quality is **downstream**: does an SFT model built on this base
produce more `kompile`-clean K? Track val/test **perplexity** here as the proxy.

## 5b. Evaluation suite (`scripts/eval_suite.py`)

Three layers, all base-vs-CPT comparable:
- **L1 perplexity** — held-out test perplexity, overall **+ per-language**.
- **L2 compile/exec** — a **held-out benchmark** (`data/benchmark/*`) of self-contained
  K definitions. The model **completes** each task's prefix (it's a base/completion
  model, not instruction-prompted); the harness then **`kompile`s** the assembled
  definition and **`krun`s** its sample programs vs expected outputs →
  **compile-rate** + **exec-correctness**.
- **L3 regression** — deprecated-syntax emission rate in completions + kompile
  error-type histogram.

```bash
# Baseline on the untrained base (run BEFORE training):
bash scripts/eval.sh --model unsloth/Qwen3-8B-Base --label base
# After CPT:
bash scripts/eval.sh --model outputs/cpt-qwen3-8b/adapter --label cpt
# Side-by-side:
uv run python scripts/eval_suite.py --compare outputs/eval/base.json outputs/eval/cpt.json
```
Reports land in `outputs/eval/<label>.json`. Note: L2 runs many `kompile`s
(1–3 min each) — it parallelizes with `--jobs`. Use `--samples K` for pass@K
(compile-rate if *any* of K samples compiles), `--skip-ppl` to skip L1,
`--ppl-max-docs N` to cap perplexity docs for speed.

---

## 6. Key config knobs (`configs/cpt.yaml`)

| Knob | Meaning | When to change |
|------|---------|----------------|
| `model.name` | base model | `unsloth/Qwen3-14B-Base` etc. to scale up |
| `model.max_seq_length` | sequence length | raise on bigger GPUs (more VRAM) |
| `mixture.k_fraction` | K vs replay | raise toward 0.8 if forgetting isn't an issue |
| `lora.r` / `lora.alpha` | adapter capacity | raise (e.g. 64) if underfitting |
| `lora.train_embeddings` | train embed/lm_head | **off** by default (K adds no new vocab; memory-heavy). Turn on only with VRAM headroom |
| `train.epochs` | passes over data | small corpus → 2–4; watch overfitting |
| `train.learning_rate` | LoRA LR | 1e-5…5e-5 typical |
| `data.extra_k_shards` | **your generated `.k`** | see §7 |

---

## 7. Adding the externally-generated big-language semantics

When the Go/Rust/Ruby/etc. `.k` semantics are ready, they slot in as extra K data:

1. Put each language's `.k` files under e.g. `data/synthetic/<lang>/`.
2. Add the paths to `data.extra_k_shards` in `configs/cpt.yaml`.
3. Re-run `make_splits.py` (it will need a small extension to read extra shards —
   currently it reads `final_manifest.jsonl`; add the shard dirs there or extend
   the loader) then `pack_dataset.py`, then `train_cpt.py`.

Because these are large, novel, idiomatic semantics, they are the highest-value K
data — they raise the corpus well past the ~5M-token public ceiling.

---

## 8. Scaling to bigger Qwen3

Change `model.name` to a bigger base. On 16 GB you can QLoRA up to ~14B (drop
`max_seq_length` to 1024 if needed). Beyond that needs a bigger GPU. Everything
else (data, mixture, splits) is model-agnostic; only the tokenizer changes, which
the pipeline handles automatically.

---

## 9. Troubleshooting

- **OOM:** lower `per_device_batch_size`, raise `grad_accum` to keep effective batch;
  lower `max_seq_length`; ensure `train_embeddings: false`.
- **Replay download fails:** `pack_dataset.py` warns and continues without that
  category; or run with `--no-replay` and add replay later.
- **unsloth downgraded torch / cuda errors:** shouldn't happen under uv (the lock
  pins the cu128 build); if the env looks off, `uv sync` restores it from `uv.lock`.
- **Val perplexity not improving:** raise `lora.r`, raise LR, or add epochs — but
  beware overfitting on this small corpus (the early-stopping guard is there for it).

---

## 10. Next phases (not in this repo yet)

- **SFT** on `(source code, K semantics)` pairs.
- **RLVR** with `kompile`/`krun` as the verifiable reward.

This CPT phase produces the K-fluent base they build on.
