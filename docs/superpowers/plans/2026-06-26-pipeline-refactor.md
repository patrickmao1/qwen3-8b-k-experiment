# Pipeline Structural Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the loose `scripts/*.py` pipeline into an importable `kcpt/` package with a typed/validated config split, folding in the correctness wins that live in the new shared modules.

**Architecture:** Introduce an installable `kcpt/` package (`config`, `paths`, `data`, `model`, `metrics`). Scripts become thin CLIs importing `kcpt`. The single `configs/cpt.yaml` splits into self-contained `configs/data.yaml` + `configs/train.yaml`, with a packed-data↔training tokenizer/seq_len guard. Migration is staged module-by-module, gated by the existing `--smoke` paths.

**Tech Stack:** Python 3.12, uv (package mode), dataclasses (stdlib config schema, no new runtime dep), pytest + ruff (dev group), HF Transformers/Trainer, unsloth, PyYAML.

## Global Constraints

- Python: `requires-python == "3.12.*"` (unchanged).
- Dependency pins stay exact (`==`); cu128 torch/torchvision from the pinned `pytorch-cu128` index; `setuptools<81` (tensorboard's `pkg_resources`).
- New runtime deps: NONE (config schema uses stdlib `dataclasses`). New dev-group deps only: `pytest`, `ruff`.
- `.sh` launchers and their documented commands (`bash scripts/{train,eval,prompt}.sh`) must keep working unchanged in interface.
- `data/`, `outputs/`, `logs/`, `.venv/` stay gitignored.
- OUT OF SCOPE (do NOT touch — these are FINDINGS.md items): document-boundary attention masking, weighting-by-duplication semantics, `save_steps`/`eval_steps` footgun, `embedding_learning_rate` wiring. Move that code verbatim; do not "fix" behavior.
- Every stage must keep the env importable (torch `2.10.0+cu128`, `cuda.is_available()`, unsloth import) — verify after the packaging change.

---

## File Structure

**New package `kcpt/`:**
- `kcpt/__init__.py` — empty marker.
- `kcpt/paths.py` — repo paths, nix `ENV`, `doc_path(row)`.
- `kcpt/config.py` — dataclass schemas + validating loaders + `snapshot_to`.
- `kcpt/data.py` — split/manifest reading, tokenization, packing, drop-counters.
- `kcpt/model.py` — `load_model()` (moved from `eval_suite`).
- `kcpt/metrics.py` — single `perplexity_from_loss()` + `corpus_perplexity()`.

**New configs:** `configs/data.yaml`, `configs/train.yaml` (replace `configs/cpt.yaml`).

**New tests:** `tests/test_data.py`, `tests/test_config.py`, `tests/test_metrics.py`.

**Modified:** `pyproject.toml`, `scripts/{make_splits,build_replay,pack_dataset,train_cpt,eval_suite,prompt}.py`, `scripts/{train,eval,prompt}.sh`, `scripts/env.sh` (new), `GUIDE.md`, `.gitignore`.

---

## Task 1: Make the project an installable package

**Files:**
- Create: `kcpt/__init__.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: an importable `kcpt` package (`uv run python -c "import kcpt"` works).

- [ ] **Step 1: Create the package marker**

`kcpt/__init__.py`:
```python
"""K-framework CPT pipeline library."""
```

- [ ] **Step 2: Switch pyproject to package mode**

In `pyproject.toml`, DELETE the line `package = false` under `[tool.uv]` (leave the rest of `[tool.uv]`). Add a build system and hatch target at the END of the file:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["kcpt"]
```

- [ ] **Step 3: Add dev dependency group + ruff config**

In `pyproject.toml`, extend `[dependency-groups]`:
```toml
[dependency-groups]
eval = ["lm-eval==0.4.12"]
dev = ["pytest>=8.0", "ruff>=0.6"]
```
Add ruff config at end of file:
```toml
[tool.ruff]
line-length = 100
[tool.ruff.lint]
select = ["E", "F", "I"]
ignore = ["E501"]
```

- [ ] **Step 4: Sync and verify the env still works**

Run: `uv sync --group dev`
Then: `uv run python -c "import kcpt, torch, unsloth; print('kcpt ok', torch.__version__, torch.cuda.is_available())"`
Expected: `kcpt ok 2.10.0+cu128 True` (unsloth banner lines are fine).

- [ ] **Step 5: Commit**

```bash
git add kcpt/__init__.py pyproject.toml uv.lock
git commit -m "Make project an installable package (kcpt) + dev tooling"
```

---

## Task 2: `kcpt/paths.py` — centralize paths and the manifest mapping

**Files:**
- Create: `kcpt/paths.py`, `tests/test_data.py` (paths portion)

**Interfaces:**
- Produces: `paths.ROOT`, `paths.DATA`, `paths.SPLITS`, `paths.CORPUS_FINAL`, `paths.PACKED`, `paths.BENCH`, `paths.OUTPUTS`, `paths.LOGS`, `paths.FINAL_MANIFEST` (str), `paths.ENV` (dict), `paths.doc_path(row: dict) -> str`.

- [ ] **Step 1: Write the failing test**

`tests/test_data.py`:
```python
from kcpt import paths

def test_doc_path_maps_repo_slashes():
    row = {"kind": "k_code", "repo": "runtimeverification/evm-semantics", "path": "evm.k"}
    p = paths.doc_path(row)
    assert p.endswith("/corpus_final/k_code/runtimeverification__evm-semantics/evm.k")

def test_env_has_nix_on_path():
    assert ".nix-profile/bin" in paths.ENV["PATH"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL (`ModuleNotFoundError: kcpt.paths`).

- [ ] **Step 3: Implement `kcpt/paths.py`**

```python
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SPLITS = os.path.join(DATA, "splits")
CORPUS_FINAL = os.path.join(DATA, "corpus_final")
PACKED = os.path.join(DATA, "packed")
BENCH = os.path.join(DATA, "benchmark")
FINAL_MANIFEST = os.path.join(DATA, "final_manifest.jsonl")
OUTPUTS = os.path.join(ROOT, "outputs")
LOGS = os.path.join(ROOT, "logs")

# kompile/krun live in the nix profile; subprocesses need it on PATH.
ENV = dict(os.environ)
ENV["PATH"] = os.path.expanduser("~/.nix-profile/bin") + ":" + ENV.get("PATH", "")


def doc_path(row):
    """Manifest row -> on-disk corpus file path."""
    return os.path.join(CORPUS_FINAL, row["kind"], row["repo"].replace("/", "__"), row["path"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_data.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add kcpt/paths.py tests/test_data.py
git commit -m "kcpt.paths: centralize repo paths + doc_path mapping"
```

---

## Task 3: `kcpt/config.py` — typed config schema + split YAMLs

**Files:**
- Create: `kcpt/config.py`, `configs/data.yaml`, `configs/train.yaml`, `tests/test_config.py`
- Delete: `configs/cpt.yaml` (after callers migrated — done in later tasks; create the new files now)

**Interfaces:**
- Produces:
  - `config.load_data_config(path=None) -> DataConfig`
  - `config.load_train_config(path=None) -> TrainConfig`
  - `config.snapshot_to(run_dir: str, *config_paths: str) -> None`
  - `DataConfig(model_name: str, max_seq_length: int, split: SplitCfg, mixture: MixtureCfg, replay_sources: dict, extra_k_shards: list)`
  - `SplitCfg(train, val, test: float; seed, by, decontaminate)`
  - `MixtureCfg(k_fraction, replay_fraction: float; replay: dict; use_doc_weights: bool)`
  - `TrainConfig(model: ModelCfg, lora: LoraCfg, train: TrainCfg)`
  - `ModelCfg(name: str, max_seq_length, load_in_4bit, dtype)`
  - `LoraCfg(r, alpha, dropout, target_modules, train_embeddings, use_gradient_checkpointing)`
  - `TrainCfg(output_dir, epochs, per_device_batch_size, grad_accum, learning_rate, lr_scheduler, warmup_ratio, weight_decay, max_grad_norm, logging_steps, eval_steps, save_steps, early_stopping_patience, seed, report_to, embedding_learning_rate, packing)`

- [ ] **Step 1: Write the new config YAMLs**

`configs/data.yaml`:
```yaml
# Data-pipeline config: drives make_splits -> build_replay -> pack_dataset.
# model_name + max_seq_length are the tokenizer/seq_len the corpus is packed FOR.
model_name: "unsloth/Qwen3-8B-Base"
max_seq_length: 2048
extra_k_shards: []          # drop-in dirs of *.k mixed as K data

split:
  train: 0.80
  val: 0.10
  test: 0.10
  seed: 1234
  by: "file"
  decontaminate: true

mixture:
  k_fraction: 0.70
  replay_fraction: 0.30
  replay:
    code: 0.70
    reasoning_math: 0.20
    general_text: 0.10
  use_doc_weights: true

replay_sources:
  code:
    dataset: "bigcode/the-stack-smol"
    split: "train"
    text_field: "content"
    interleave_dirs: ["data/python","data/java","data/javascript","data/typescript",
      "data/c","data/c++","data/c-sharp","data/go","data/rust","data/ruby","data/php",
      "data/scala","data/haskell","data/lua","data/shell","data/sql"]
  reasoning_math:
    dataset: "open-r1/OpenR1-Math-220k"
    split: "train"
    text_field: "problem"
  general_text:
    dataset: "HuggingFaceFW/fineweb-edu"
    subset: "sample-10BT"
    split: "train"
    text_field: "text"
```

`configs/train.yaml`:
```yaml
# CPT recipe: drives train_cpt. model.name/max_seq_length must match what the
# packed data was built for (asserted at train time).
model:
  name: "unsloth/Qwen3-8B-Base"
  max_seq_length: 2048
  load_in_4bit: true
  dtype: null

lora:
  r: 32
  alpha: 32
  dropout: 0.0
  target_modules: ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
  train_embeddings: false
  use_gradient_checkpointing: "unsloth"

train:
  output_dir: "outputs/cpt-qwen3-8b"
  epochs: 3
  per_device_batch_size: 2
  grad_accum: 8
  learning_rate: 3.0e-5
  embedding_learning_rate: 3.0e-6
  lr_scheduler: "cosine"
  warmup_ratio: 0.05
  weight_decay: 0.01
  max_grad_norm: 1.0
  logging_steps: 5
  eval_steps: 50
  save_steps: 100
  early_stopping_patience: 3
  seed: 1234
  report_to: "tensorboard"
  packing: true
```

- [ ] **Step 2: Write the failing test**

`tests/test_config.py`:
```python
import pytest
from kcpt import config

def test_load_data_config_valid():
    c = config.load_data_config()
    assert c.model_name == "unsloth/Qwen3-8B-Base"
    assert c.max_seq_length == 2048
    assert c.split.train == 0.80
    assert c.mixture.k_fraction == 0.70
    assert "code" in c.replay_sources

def test_load_train_config_valid():
    c = config.load_train_config()
    assert c.model.name == "unsloth/Qwen3-8B-Base"
    assert c.lora.r == 32
    assert c.train.report_to == "tensorboard"

def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("model:\n  name: x\n  bogus_key: 1\nlora:\n  r: 1\n  alpha: 1\n  dropout: 0.0\n  target_modules: []\ntrain:\n  output_dir: o\n  epochs: 1\n  per_device_batch_size: 1\n  grad_accum: 1\n  learning_rate: 1.0\n")
    with pytest.raises(ValueError, match="bogus_key"):
        config.load_train_config(str(p))

def test_missing_required_key_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("model:\n  name: x\nlora:\n  r: 1\n")  # missing train, lora fields
    with pytest.raises(ValueError):
        config.load_train_config(str(p))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL (`ModuleNotFoundError: kcpt.config`).

- [ ] **Step 4: Implement `kcpt/config.py`**

```python
import os
import shutil
from dataclasses import dataclass, field, fields, is_dataclass, MISSING

import yaml

from kcpt import paths


@dataclass
class SplitCfg:
    train: float
    val: float
    test: float
    seed: int = 1234
    by: str = "file"
    decontaminate: bool = True


@dataclass
class MixtureCfg:
    k_fraction: float
    replay_fraction: float
    replay: dict
    use_doc_weights: bool = True


@dataclass
class DataConfig:
    model_name: str
    max_seq_length: int
    split: SplitCfg
    mixture: MixtureCfg
    replay_sources: dict
    extra_k_shards: list = field(default_factory=list)


@dataclass
class ModelCfg:
    name: str
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    dtype: object = None


@dataclass
class LoraCfg:
    r: int
    alpha: int
    dropout: float
    target_modules: list
    train_embeddings: bool = False
    use_gradient_checkpointing: str = "unsloth"


@dataclass
class TrainCfg:
    output_dir: str
    epochs: int
    per_device_batch_size: int
    grad_accum: int
    learning_rate: float
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 5
    eval_steps: int = 50
    save_steps: int = 100
    early_stopping_patience: int = 3
    seed: int = 1234
    report_to: str = "tensorboard"
    embedding_learning_rate: float = 3.0e-6
    packing: bool = True


@dataclass
class TrainConfig:
    model: ModelCfg
    lora: LoraCfg
    train: TrainCfg


def _build(cls, d, where):
    if not isinstance(d, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(d).__name__}")
    allowed = {f.name for f in fields(cls)}
    unknown = set(d) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)} (allowed: {sorted(allowed)})")
    kwargs = {}
    for f in fields(cls):
        if f.name in d:
            v = d[f.name]
            kwargs[f.name] = _build(f.type, v, f"{where}.{f.name}") if is_dataclass(f.type) else v
        elif f.default is MISSING and f.default_factory is MISSING:
            raise ValueError(f"{where}: missing required key '{f.name}'")
    return cls(**kwargs)


def load_data_config(path=None):
    path = path or os.path.join(paths.ROOT, "configs", "data.yaml")
    return _build(DataConfig, yaml.safe_load(open(path)), "data")


def load_train_config(path=None):
    path = path or os.path.join(paths.ROOT, "configs", "train.yaml")
    return _build(TrainConfig, yaml.safe_load(open(path)), "train")


def snapshot_to(run_dir, *config_paths):
    """Copy resolved config files into a run's output dir for provenance."""
    os.makedirs(run_dir, exist_ok=True)
    for p in config_paths:
        shutil.copy(p, os.path.join(run_dir, os.path.basename(p)))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit (keep `configs/cpt.yaml` for now — callers migrate next)**

```bash
git add kcpt/config.py configs/data.yaml configs/train.yaml tests/test_config.py
git commit -m "kcpt.config: typed schema + split data/train YAMLs"
```

---

## Task 4: `kcpt/data.py` — split reading, tokenization, packing (fail-loud)

**Files:**
- Create: `kcpt/data.py`
- Modify: `tests/test_data.py` (add packing/weight tests)

**Interfaces:**
- Consumes: `paths.doc_path`, `paths.SPLITS`.
- Produces:
  - `data.read_split(name: str) -> list[dict]`
  - `data.weighted_copies(weight: float, rng: random.Random) -> int`
  - `data.pack_token_lists(token_lists: list[list[int]], seq_len: int, eos_id: int) -> list[list[int]]`
  - `data.load_split_token_lists(name, tok, *, max_k=0, use_weights=False, rng=None) -> tuple[list[list[int]], int]` returns (token_lists, total_tokens); raises `RuntimeError` if all docs dropped.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_data.py`:
```python
import random
from kcpt import data

def test_pack_token_lists_chunks_and_inserts_eos():
    lists = [[1, 2, 3], [4, 5]]
    blocks = data.pack_token_lists(lists, seq_len=3, eos_id=0)
    # stream = 1,2,3,0,4,5,0 -> blocks of 3: [1,2,3],[0,4,5]; trailing [0] dropped
    assert blocks == [[1, 2, 3], [0, 4, 5]]

def test_pack_token_lists_drops_partial_final_block():
    blocks = data.pack_token_lists([[1, 2]], seq_len=4, eos_id=0)
    assert blocks == []  # 1,2,0 < 4

def test_weighted_copies_floor_plus_fractional():
    rng = random.Random(0)
    counts = [data.weighted_copies(1.5, rng) for _ in range(1000)]
    assert all(c in (1, 2) for c in counts)
    assert 1 in counts and 2 in counts  # both branches hit

def test_weighted_copies_integer_weight_exact():
    rng = random.Random(0)
    assert all(data.weighted_copies(2.0, rng) == 2 for _ in range(50))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL (`ModuleNotFoundError: kcpt.data`).

- [ ] **Step 3: Implement `kcpt/data.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_data.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add kcpt/data.py tests/test_data.py
git commit -m "kcpt.data: split/pack/weight helpers with fail-loud drop guard"
```

---

## Task 5: `kcpt/metrics.py` — single perplexity definition

**Files:**
- Create: `kcpt/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `metrics.perplexity_from_loss(loss: float, clamp: float = 20.0) -> float`
  - `metrics.corpus_perplexity(model, tok, rows, max_seq_length, *, doc_path_fn) -> dict` with keys `overall_perplexity`, `per_language` (moved from `eval_suite.perplexity`; signature change: takes `rows` + `doc_path_fn` instead of reading globals).

- [ ] **Step 1: Write the failing test**

`tests/test_metrics.py`:
```python
import math
from kcpt import metrics

def test_perplexity_from_loss_basic():
    assert math.isclose(metrics.perplexity_from_loss(0.0), 1.0)
    assert math.isclose(metrics.perplexity_from_loss(1.0), math.e, rel_tol=1e-6)

def test_perplexity_from_loss_clamps():
    # huge loss clamps to exp(20), not inf
    assert metrics.perplexity_from_loss(1e9) == math.exp(20)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL (`ModuleNotFoundError: kcpt.metrics`).

- [ ] **Step 3: Implement `kcpt/metrics.py`**

```python
import collections
import math
import os


def perplexity_from_loss(loss, clamp=20.0):
    return math.exp(min(loss, clamp))


def corpus_perplexity(model, tok, rows, max_seq_length, *, doc_path_fn, max_docs=0):
    """Per-document held-out perplexity, overall + per repo. Pure of globals:
    caller supplies rows and a doc_path_fn(row)->path."""
    import torch

    if max_docs:
        rows = rows[:max_docs]
    by = collections.defaultdict(lambda: [0.0, 0])
    tot = [0.0, 0]
    for r in rows:
        p = doc_path_fn(r)
        try:
            text = open(p, encoding="utf-8", errors="replace").read()
        except FileNotFoundError:
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        for i in range(0, len(ids), max_seq_length):
            chunk = ids[i:i + max_seq_length]
            if len(chunk) < 2:
                continue
            t = torch.tensor([chunk], device=model.device)
            with torch.no_grad():
                loss = model(t, labels=t).loss.item()
            n = len(chunk) - 1
            by[r["repo"]][0] += loss * n
            by[r["repo"]][1] += n
            tot[0] += loss * n
            tot[1] += n
    per_lang = {repo: round(perplexity_from_loss(s / n), 3) for repo, (s, n) in by.items() if n}
    overall = round(perplexity_from_loss(tot[0] / max(tot[1], 1)), 3)
    return {"overall_perplexity": overall, "per_language": dict(sorted(per_lang.items()))}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add kcpt/metrics.py tests/test_metrics.py
git commit -m "kcpt.metrics: single perplexity definition"
```

---

## Task 6: `kcpt/model.py` — shared model loading

**Files:**
- Create: `kcpt/model.py`

**Interfaces:**
- Produces: `model.load_model(model_id: str, max_seq_length: int) -> (model, tokenizer)` (moved verbatim from `eval_suite.load_model`).

- [ ] **Step 1: Implement `kcpt/model.py`** (no unit test — requires GPU; covered by the eval/prompt smokes in Task 9)

```python
def load_model(model_id, max_seq_length):
    """Load a (base or adapter) model in 4-bit for inference via unsloth."""
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    return model, tok
```

- [ ] **Step 2: Verify import**

Run: `uv run python -c "from kcpt.model import load_model; print('ok')"`
Expected: `ok` (unsloth banner fine).

- [ ] **Step 3: Commit**

```bash
git add kcpt/model.py
git commit -m "kcpt.model: shared 4-bit load_model"
```

---

## Task 7: Repoint data-pipeline scripts (`make_splits`, `build_replay`, `pack_dataset`)

**Files:**
- Modify: `scripts/make_splits.py`, `scripts/build_replay.py`, `scripts/pack_dataset.py`

**Interfaces:**
- Consumes: `kcpt.config.load_data_config`, `kcpt.paths`, `kcpt.data`.
- Produces: `data/packed/stats.json` now includes `tokenizer` and `seq_len` keys.

- [ ] **Step 1: Migrate `scripts/make_splits.py`**

Replace its `load_cfg()` and `path()` and the config access. At top, replace the local helpers with:
```python
from kcpt import paths
from kcpt.config import load_data_config

def path(r):
    return paths.doc_path(r)
```
Replace `MAN = ...`/`FINAL = ...`/`OUT = ...` module constants with `paths.FINAL_MANIFEST`, `paths.CORPUS_FINAL`, `paths.SPLITS`. In `main()`, replace `cfg = load_cfg()` with:
```python
dc = load_data_config()
cfg = {"train": dc.split.train, "val": dc.split.val, "test": dc.split.test,
       "seed": dc.split.seed, "decontaminate": dc.split.decontaminate}
```
(Keep the rest of `make_splits` logic unchanged.)

- [ ] **Step 2: Run make_splits on real data**

Run: `uv run python scripts/make_splits.py`
Expected: prints `split N docs -> train ... val ... test ...`; `data/splits/{train,val,test}.jsonl` regenerated.

- [ ] **Step 3: Migrate `scripts/build_replay.py`**

Remove its local `load_cfg`; keep `iter_category` and `replay_budgets` (these are imported by pack_dataset). Change `replay_budgets(cfg, ...)` to accept the dataclass:
```python
def replay_budgets(mixture, k_train_tokens):
    total = (mixture.replay_fraction / mixture.k_fraction) * k_train_tokens
    return {k: total * mixture.replay[k] for k in mixture.replay}
```
In `main()`, replace config load with `dc = load_data_config()`, tokenizer from `dc.model_name`, specs from `dc.replay_sources`, and call `replay_budgets(dc.mixture, args.k_tokens)`. Add `from kcpt.config import load_data_config` and `from kcpt import paths`; use `paths` for `OUT = os.path.join(paths.DATA, "replay")`.

- [ ] **Step 4: Migrate `scripts/pack_dataset.py`**

Delete the `sys.path.insert`, local `load_cfg`, `doc_path`, `read_split`, `weighted_copies`, `pack_token_lists`. Import instead:
```python
from kcpt import paths
from kcpt.config import load_data_config
from kcpt.data import pack_token_lists, load_split_token_lists
from build_replay import iter_category, replay_budgets  # build_replay still a script-local import
```
Note: `build_replay` is still under `scripts/`. To import it from `pack_dataset`, move the shared replay streamer too — SIMPLEST: keep `from build_replay import ...` working by adding `from scripts.build_replay import ...`? Instead, move `iter_category`/`replay_budgets` into `kcpt/data.py` is out of scope; keep them in `build_replay.py` and import via the package path: add `import sys, os; sys.path.insert(0, os.path.dirname(__file__))` is what we're removing. RESOLUTION: in Task 7 Step 3, ALSO expose them by importing build_replay through a tiny shim — add to `kcpt/data.py`:
```python
def replay_iter(name, spec, budget, tok):
    from scripts.build_replay import iter_category  # scripts dir importable? no.
```
This is fragile. FINAL DECISION (do this): move `iter_category` and `replay_budgets` into `kcpt/data.py` (they are data helpers), and have `scripts/build_replay.py` import them from `kcpt.data`. Then `pack_dataset` imports from `kcpt.data` too. Update Task 7 to:
  - Move `iter_category` + `replay_budgets(mixture, k)` into `kcpt/data.py`.
  - `scripts/build_replay.py` becomes a thin CLI: `from kcpt.data import iter_category, replay_budgets`.
  - `scripts/pack_dataset.py`: `from kcpt.data import iter_category, replay_budgets, pack_token_lists, load_split_token_lists`.

In `pack_dataset.main()`, use `dc = load_data_config()`, `tok = AutoTokenizer.from_pretrained(dc.model_name)`, `seq_len = dc.max_seq_length`, replay specs `dc.replay_sources`, budgets `replay_budgets(dc.mixture, k_tokens["train"])`. Replace the K-split loop with calls to `load_split_token_lists(name, tok, max_k=max_k, use_weights=(dc.mixture.use_doc_weights and name=="train"), rng=rng)`. When writing `stats.json`, ADD:
```python
stats["tokenizer"] = dc.model_name
stats["seq_len"] = seq_len
```

- [ ] **Step 5: Add `iter_category`/`replay_budgets` to `kcpt/data.py` + a budget test**

Append to `kcpt/data.py` the `iter_category` function (verbatim from `scripts/build_replay.py` lines 19-49) and:
```python
def replay_budgets(mixture, k_train_tokens):
    total = (mixture.replay_fraction / mixture.k_fraction) * k_train_tokens
    return {k: total * mixture.replay[k] for k in mixture.replay}
```
Append to `tests/test_data.py`:
```python
from kcpt.config import MixtureCfg

def test_replay_budgets_proportional():
    mix = MixtureCfg(k_fraction=0.7, replay_fraction=0.3,
                     replay={"code": 0.7, "reasoning_math": 0.2, "general_text": 0.1})
    b = data.replay_budgets(mix, 1_000_000)
    total = (0.3 / 0.7) * 1_000_000
    assert math.isclose(sum(b.values()), total, rel_tol=1e-9)
    assert math.isclose(b["code"], total * 0.7, rel_tol=1e-9)
```

- [ ] **Step 6: Run pack smoke + unit tests**

Run: `uv run pytest tests/ -v`
Expected: PASS (all).
Run: `uv run python scripts/pack_dataset.py --smoke`
Expected: prints K split token counts + `packed train/val/test`; `data/packed/stats.json` now has `tokenizer` + `seq_len`.

- [ ] **Step 7: Commit**

```bash
git add kcpt/data.py scripts/make_splits.py scripts/build_replay.py scripts/pack_dataset.py tests/test_data.py
git commit -m "Repoint data pipeline to kcpt; record tokenizer/seq_len in stats"
```

---

## Task 8: Repoint `train_cpt.py` (config split + snapshot + guard + shared perplexity)

**Files:**
- Modify: `scripts/train_cpt.py`

**Interfaces:**
- Consumes: `kcpt.config.load_train_config`, `kcpt.metrics.perplexity_from_loss`, `kcpt.config.snapshot_to`, `kcpt.paths`.

- [ ] **Step 1: Migrate imports + config access**

In `scripts/train_cpt.py`: delete local `load_cfg`. Add:
```python
from kcpt import paths
from kcpt.config import load_train_config, snapshot_to
from kcpt.metrics import perplexity_from_loss
```
Replace `cfg = load_cfg(); m, lo, tr = cfg["model"], cfg["lora"], cfg["train"]` with:
```python
cfg = load_train_config()
m, lo, tr = cfg.model, cfg.lora, cfg.train
```
Then replace every `m["..."]`/`lo["..."]`/`tr["..."]` access with attribute access (`m.name`, `m.max_seq_length`, `m.load_in_4bit`, `m.dtype`, `lo.r`, `lo.alpha`, `lo.dropout`, `lo.target_modules`, `lo.train_embeddings`, `lo.use_gradient_checkpointing`, `tr.output_dir`, `tr.epochs`, `tr.per_device_batch_size`, `tr.grad_accum`, `tr.learning_rate`, `tr.lr_scheduler`, `tr.warmup_ratio`, `tr.weight_decay`, `tr.max_grad_norm`, `tr.logging_steps`, `tr.eval_steps`, `tr.save_steps`, `tr.early_stopping_patience`, `tr.seed`, `tr.report_to`). Update `PACKED` to `paths.PACKED`.

- [ ] **Step 2: Replace the CPTTrainer perplexity helper**

In the `CPTTrainer.log` override, replace `math.exp(min(logs["eval_loss"], 20))` with `perplexity_from_loss(logs["eval_loss"])`. Likewise the test-metrics block: `res["test_perplexity"] = perplexity_from_loss(res["eval_loss"])`.

- [ ] **Step 3: Add the packed-data guard + config snapshot**

After loading `train_ds` and before building `targs`, insert:
```python
    import json
    stats = json.load(open(os.path.join(paths.PACKED, "stats.json")))
    if not args.smoke:
        assert stats.get("tokenizer") == m.name, (
            f"packed data tokenizer {stats.get('tokenizer')} != train model {m.name}; re-run pack_dataset")
        assert stats.get("seq_len") == m.max_seq_length, (
            f"packed seq_len {stats.get('seq_len')} != {m.max_seq_length}")
```
After `trainer.train()` and inside the `if not args.smoke:` block, add a snapshot of both configs into the run dir:
```python
        snapshot_to(tr.output_dir, os.path.join(paths.ROOT, "configs", "train.yaml"),
                    os.path.join(paths.ROOT, "configs", "data.yaml"))
```

- [ ] **Step 4: Run the train smoke**

Run: `bash scripts/train.sh --smoke`
Expected: loads 4-bit model, 8 steps, eval, `done. peak VRAM: ~10 GB`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_cpt.py
git commit -m "Repoint train_cpt to kcpt config/metrics; add tokenizer guard + config snapshot"
```

---

## Task 9: Repoint `eval_suite.py` and `prompt.py`

**Files:**
- Modify: `scripts/eval_suite.py`, `scripts/prompt.py`

**Interfaces:**
- Consumes: `kcpt.model.load_model`, `kcpt.metrics.corpus_perplexity`, `kcpt.paths`.

- [ ] **Step 1: Migrate `scripts/eval_suite.py`**

Delete its local `load_model` and `perplexity`. Add:
```python
from kcpt import paths
from kcpt.model import load_model
from kcpt.metrics import corpus_perplexity
```
Replace module constants `BENCH/TEST/FINAL/OUTDIR/ENV` with `paths.BENCH`, `os.path.join(paths.SPLITS, "test.jsonl")`, `paths.CORPUS_FINAL`, `os.path.join(paths.OUTPUTS, "eval")`, `paths.ENV`. Replace the `perplexity(model, tok, max_seq_length, max_docs)` call in `main()` with:
```python
    import json as _json
    rows = [_json.loads(l) for l in open(os.path.join(paths.SPLITS, "test.jsonl"))]
    L1 = {} if args.skip_ppl else corpus_perplexity(
        model, tok, rows, args.max_seq_length, doc_path_fn=paths.doc_path, max_docs=args.ppl_max_docs)
```

- [ ] **Step 2: Run eval on a tiny slice**

Run: `bash scripts/eval.sh --model unsloth/Qwen3-8B-Base --label _smoke --ppl-max-docs 2 --skip-ppl` then without `--skip-ppl` for 2 docs:
`bash scripts/eval.sh --model outputs/cpt-qwen3-8b/adapter --label _smoke --ppl-max-docs 2`
Expected: writes `outputs/eval/_smoke.json` with an `L1.overall_perplexity` number; no crash.

- [ ] **Step 3: Migrate `scripts/prompt.py`**

Replace `from eval_suite import load_model` with `from kcpt.model import load_model`. (Everything else unchanged.)

- [ ] **Step 4: Run the prompt smoke**

Run: `bash scripts/prompt.sh -f prompts/imp.k --max-new 64`
Expected: prints a K completion; exit 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_suite.py scripts/prompt.py
git commit -m "Repoint eval_suite + prompt to kcpt.model/metrics"
```

---

## Task 10: Retire `configs/cpt.yaml`, shared `env.sh`, `logs/` tidy, docs

**Files:**
- Delete: `configs/cpt.yaml`
- Create: `scripts/env.sh`
- Modify: `scripts/{train,eval,prompt}.sh`, `.gitignore`, `GUIDE.md`

**Interfaces:** none (cleanup).

- [ ] **Step 1: Confirm no remaining references to cpt.yaml**

Run: `grep -rn "cpt.yaml" scripts/ kcpt/ GUIDE.md`
Expected: no matches. If any remain, fix them before deleting. Then: `git rm configs/cpt.yaml`.

- [ ] **Step 2: Create shared `scripts/env.sh`**

```bash
# Shared env for the launchers: nix tools + venv on PATH, C compiler for Triton JIT.
export PATH="$HOME/.nix-profile/bin:$HOME/.local/bin:$PATH"
export CC=gcc CXX=g++
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
command -v cc >/dev/null || { echo "ERROR: no C compiler (cc). Run: nix profile install nixpkgs#gcc"; exit 1; }
```

- [ ] **Step 3: Slim the three launchers to source it**

Each of `scripts/train.sh`, `scripts/eval.sh`, `scripts/prompt.sh` becomes (example `train.sh`):
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(dirname "$0")/env.sh"
exec uv run python scripts/train_cpt.py "$@"
```
(eval.sh: also keep its `command -v kompile` check after sourcing env.sh; prompt.sh: `scripts/prompt.py`.)

- [ ] **Step 4: `logs/` tidy**

Add `logs/` to `.gitignore` (new line). In `GUIDE.md`, change the nohup example `> data/train.log` to `> logs/train.log` and add `mkdir -p logs` before it. Update the "## 0 Prerequisites" config bullet to mention `configs/data.yaml` + `configs/train.yaml` instead of `cpt.yaml`.

- [ ] **Step 5: Full regression — smokes + tests + ruff**

Run: `uv run pytest tests/ -v` → all pass.
Run: `uv run ruff check kcpt/ scripts/` → fix any reported issues, re-run until clean.
Run: `bash scripts/prompt.sh -f prompts/imp.k --max-new 32` → exits 0 (final end-to-end through the new launcher + env.sh).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Retire cpt.yaml; shared env.sh; logs/ tidy; ruff clean; docs"
```

---

## Self-Review

**Spec coverage:**
- kcpt package (config/paths/data/model/metrics) → Tasks 1–6. ✓
- Typed/validated config → Task 3 (incl. unknown/missing-key tests). ✓
- Config split data/train → Task 3; cpt.yaml retired Task 10. ✓
- Per-run snapshot → Task 8 Step 3. ✓
- One perplexity definition → Task 5; consumed Tasks 8–9. ✓
- Fail-loud on dropped docs → Task 4 (`load_split_token_lists` RuntimeError). ✓
- Tokenizer/seq_len guard → Task 7 (record) + Task 8 (assert). ✓
- Thin entrypoints + env.sh → Task 10. ✓
- logs/ tidy → Task 10. ✓
- ruff + pytest + unit tests → Tasks 1, 2, 3, 4, 5, 7, 10. ✓
- Staged migration with smokes → task order + smoke steps. ✓
- OUT OF SCOPE items untouched → enforced by "move verbatim" notes. ✓

**Placeholder scan:** Task 7 originally had an unresolved import path for `build_replay`; resolved in-task by moving `iter_category`/`replay_budgets` into `kcpt/data.py` (Task 7 Steps 4–5). No other placeholders.

**Type consistency:** `load_split_token_lists` returns `(lists, total)` — consumed as such in pack_dataset (Task 7). `replay_budgets(mixture, k)` takes the `MixtureCfg` dataclass consistently in Tasks 5/7. `perplexity_from_loss`/`corpus_perplexity` names match across Tasks 5/8/9. `corpus_perplexity(..., doc_path_fn=paths.doc_path)` matches its definition.
