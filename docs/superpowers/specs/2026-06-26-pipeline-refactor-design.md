# Pipeline Structural Refactor — Design

Date: 2026-06-26
Status: proposed (awaiting review → implementation plan)

## Goal

Consolidate the loose `scripts/*.py` pipeline into a small importable package with a
typed config, eliminating duplication and `sys.path` hacks, and fold in the
correctness wins that naturally live in the new shared modules. Pure-algorithm
training fixes stay in `FINDINGS.md` (out of scope here).

## Scope

**In scope (structural + "free" correctness):**
- Extract a `kcpt/` installable package; repoint scripts to import it.
- Typed/validated config schema (fail fast on typos).
- Split the single `cpt.yaml` into `configs/data.yaml` + `configs/train.yaml`.
- Per-run config snapshotting (provenance).
- One `perplexity()` definition shared by trainer + eval suite.
- Fail-loud on dropped corpus docs (counters + tolerance assert).
- Packed-data ↔ training tokenizer/seq_len consistency guard.
- `data/` vs `logs/` tidy.
- `ruff` + `pytest` dev tooling and unit tests on pure functions.

**Out of scope (→ FINDINGS.md, separate pass):**
- Document-boundary attention masking in packing (#1).
- Replacing weighting-by-duplication (#2).
- `save_steps`/`eval_steps` `load_best_model_at_end` footgun (#5).
- Dead `embedding_learning_rate` wiring (#3).

## Architecture

### Approach: installable `kcpt/` package (chosen over `sys.path` bootstrap / `src/` layout)

Flip the uv project to a real package (`package = true` + build backend); `uv sync`
installs `kcpt` editable. All scripts/tests use clean `from kcpt.* import ...`. No
path hacks; uniform, testable imports.

### `kcpt/` modules

- `config.py` — dataclass schemas `DataConfig` / `TrainConfig`; `load_data_config()`
  / `load_train_config()` validate on load (clear error on unknown/missing keys);
  `snapshot_to(run_dir)` copies the resolved config into the run's output dir.
- `paths.py` — `ROOT`, data/output/log dirs, the nix `ENV`, and the canonical
  manifest-row→filepath mapping `doc_path(row)` (currently duplicated ×2).
- `data.py` — split/manifest reading, tokenization, `pack_token_lists`,
  `weighted_copies`; drop-counters that assert non-empty / within tolerance.
- `model.py` — `load_model()` (currently buried in `eval_suite.py`); shared by
  train/eval/prompt.
- `metrics.py` — the single `perplexity()` used by trainer and eval suite.

### Thin entrypoints

`scripts/*.py` remain the CLIs (arg-parsing + calls into `kcpt`). The `.sh`
launchers are unchanged, so documented commands still work. A shared
`scripts/env.sh` DRYs the PATH/`CC`/alloc-conf exports.

## Config split

Two self-contained YAMLs in `configs/`:

- `configs/data.yaml` — `model.name` + `max_seq_length` (the tokenizer/seq_len the
  data is packed FOR), `split`, `mixture`, `replay_sources`, corpus paths.
  Consumed by `make_splits` → `build_replay` → `pack_dataset`.
- `configs/train.yaml` — `model` (base, 4-bit, dtype, seq_len), `lora`, `train`.
  Consumed by `train_cpt`.

**Overlap handling — self-contained + guard (standard for Axolotl/torchtune-style
finetuning; composition deferred):** `model.name` + `max_seq_length` appear in both.
`pack_dataset` records `(tokenizer_name, seq_len)` into `data/packed/stats.json`;
`train_cpt` asserts the packed data matches its `model.name`/`max_seq_length` before
training — catching "trained on mismatched-tokenizer data" early.

When SFT/RLVR arrive they get their own `configs/<phase>.yaml` (or per-phase subdirs
introduced then, not pre-built now).

## Directory / data

- New `logs/` (gitignored) for run logs currently under `data/` (`train.log`, etc.).
- Corpus / packed / manifest layout unchanged (it's fine).
- New: `kcpt/`, `tests/`, `docs/superpowers/specs/`.

## Testing & tooling

- `dev` dependency group: `pytest`, `ruff`.
- `tests/` unit tests on the now-importable pure functions: `pack_token_lists`,
  `weighted_copies`, shingling/decontam overlap math, config validation (good +
  bad configs), `perplexity`.
- `ruff` config in `pyproject.toml`.

## Migration & verification (staged, not big-bang)

Touches the whole pipeline, so extract one module at a time and verify before moving
on, using the existing `--smoke` paths as the regression net:

1. Scaffold `kcpt/` + make package installable; `uv sync`; confirm env still imports
   (torch+cu128, unsloth).
2. `paths.py` + `config.py` (with the split YAMLs + schema); repoint all `load_cfg`
   call sites; run `make_splits` on real data.
3. `data.py`; repoint `pack_dataset`/`build_replay`; `pack_dataset --smoke`.
4. `model.py` + `metrics.py`; repoint `train_cpt`/`eval_suite`/`prompt`;
   `train.sh --smoke`; `eval.sh` on a tiny slice; `prompt.sh -f prompts/imp.k`.
5. `logs/` move, `ruff`, `tests/`; `pytest` green.

Each stage is independently committable.

## Risks

- `package = true` build change could perturb the fragile cu128/unsloth env — verified
  by an import smoke right after step 1; rollback is reverting the pyproject change.
- Hidden behavioral coupling when splitting configs — the consistency guard + per-stage
  smokes catch mismatches.
