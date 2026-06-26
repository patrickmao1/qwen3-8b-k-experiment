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
