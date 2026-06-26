# Code & Training Review — Findings

A senior-ML-engineer review of the CPT pipeline (data → pack → train → eval).
Worst-first within each section. File:line references point at the offending code.

> Overall: above-average for a research repo — execution-based eval (kompile/krun),
> a replay mixture for forgetting, decontaminated K-only held-out splits, early
> stopping, config-driven pipeline. The items below are the real problems.

## Priority punch list
1. Document-boundary masking in packing (#1) — quality bug affecting every step.
2. Stop weighting-by-duplication (#2) — directly feeds the observed overfitting.
3. Unify the val perplexity metric (#4) so early stopping optimizes what we report.
4. Add pipeline unit tests (#7) and fail loud on dropped data (#8).
5. Fix the `embedding_lr` / `save_steps` footguns (#3, #5) before they bite.

---

## Training methodology

### 1. Packing has cross-document attention contamination (most important)
`pack_token_lists` (scripts/pack_dataset.py:39) concatenates multiple docs into each
2048-token block with only an EOS between them, and the collator sets
`attention_mask = torch.ones_like(ids)` (scripts/train_cpt.py:32). So every token in
doc B attends to doc A, and `position_ids` never reset across boundaries. The model is
trained on cross-document dependencies that don't exist; the noise is proportionally
larger on a small corpus.
**Fix:** block-diagonal (intra-doc) attention + per-doc `position_ids` reset
(FlashAttention varlen / unsloth document masking). At minimum mask labels at the
EOS->next-doc transition.

### 2. "Weighting" by physically duplicating documents -> memorization
`weighted_copies` (scripts/pack_dataset.py:31) realizes per-doc weights as `floor(w)`
literal copies of the token list. With 3 epochs on a ~4-6M-token niche corpus, some
docs are seen many times verbatim — a recipe for the overfitting/repetition already
observed.
**Fix:** loss-scaling or temperature-based sampling without replacement instead of hard
duplication; if duplicating, ensure copies don't co-occur in the same packed block.

### 3. Dead `embedding_learning_rate` config — latent bug
configs/cpt.yaml:89 advertises a separate (lower) embedding LR, but scripts/train_cpt.py
uses plain `TrainingArguments` with a single `learning_rate` (line 75) and never reads
`embedding_learning_rate`. If `train_embeddings: true` is ever set, embeddings train at
the full `3e-5`, contrary to the config's promise.
**Fix:** use unsloth's `UnslothTrainingArguments(embedding_learning_rate=...)`, or drop
the config key.

---

## Eval integrity

### 4. Two non-comparable perplexity definitions — early stopping uses the worse one
The trainer's `eval_loss` (-> early stopping / best-model selection) is computed over
*packed, cross-doc, EOS-joined* blocks (scripts/train_cpt.py:98; val_ds is packed). But
`eval_suite.perplexity()` re-reads raw files and chunks **per document** — the cleaner
number actually reported. Early stopping optimizes the contaminated metric.
**Fix:** make the trainer's val perplexity use the same per-doc definition, or at least
document that they differ.

### 5. `eval_steps=50` != `save_steps=100` undermines `load_best_model_at_end`
With `load_best_model_at_end=True` (scripts/train_cpt.py:81), the best checkpoint is
chosen across evals at 50,100,150,... but checkpoints only exist at 100,200,.... A best
eval on an odd step (e.g. 150) was never saved, so a coarser "best" is silently loaded.
**Fix:** set `save_steps == eval_steps` (or a clean multiple where the best can always be
saved).

### 6. Perplexity clamp hides divergence
`math.exp(min(loss, 20))` (scripts/train_cpt.py:37,99; also eval_suite) silently caps; a
diverging run (loss >> 20) reads as a flat ppl~=4.8e8 instead of an alarm.
**Fix:** keep the display guard but log the raw loss alongside.

---

## Code & engineering

### 7. No tests on the data pipeline (biggest engineering gap)
The only "test" is `--smoke` (a runtime path check). The functions most likely to
silently corrupt a run — `weighted_copies`, `pack_token_lists`, `shingles`/decontam
overlap math, mixture-fraction realization — are pure and trivially unit-testable, and
untested data bugs are the most expensive kind.
**Fix:** a handful of `pytest` cases over these pure functions.

### 8. Silent data-dropping via broad excepts
`except FileNotFoundError: continue` (scripts/pack_dataset.py:76, scripts/make_splits.py:51)
and `except Exception` per replay category (scripts/pack_dataset.py:104) will run on half
a corpus with no error.
**Fix:** count drops and assert non-empty / within tolerance — a path typo shouldn't yield
a quietly-undertrained model.

### 9. `load_cfg()` reimplemented in 4 files; config unvalidated
train_cpt, pack_dataset, build_replay, make_splits each define their own YAML loader, and
the schema is implicit — a typo in cpt.yaml surfaces as a `KeyError` deep into a GPU run.
**Fix:** one validated schema (`dataclass`/pydantic `Config`) in a shared module; import it
everywhere. Also removes the `sys.path.insert` sibling-import hack
(scripts/pack_dataset.py:17).

### 10. No experiment tracking
`report_to="none"` (scripts/train_cpt.py:83) — training curves live only in a redirected
log file.
**Fix:** at least `report_to="tensorboard"`; ideally W&B, especially heading into
CPT->SFT->RLVR.

### Minor
- Bare `open()` without context managers throughout (relies on refcount GC).
- Magic numbers hardcoded rather than in config: `SHINGLE=13`, `OVERLAP_THRESH=0.50`
  (scripts/make_splits.py:21-22), replay smoke budget `20000` (scripts/pack_dataset.py:97).
- Soft reproducibility: replay streams from HF without a pinned snapshot/revision, so
  re-running the packer can pull different replay docs.
