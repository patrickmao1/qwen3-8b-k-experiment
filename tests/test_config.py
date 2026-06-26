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
