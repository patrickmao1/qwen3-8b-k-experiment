#!/usr/bin/env bash
# Install the ML stack into .venv, protecting the cu128/Blackwell torch build.
set -uo pipefail
cd /home/patrickmao/repos/ft
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
PY=.venv/bin/python
CU128="https://download.pytorch.org/whl/cu128"

echo "### 1. core libs"
uv pip install --python $PY numpy pyyaml sentencepiece "transformers>=4.51" \
    datasets accelerate peft trl "huggingface_hub[hf_transfer]"

echo "### 2. bitsandbytes (4-bit / QLoRA)"
uv pip install --python $PY bitsandbytes

echo "### 3. unsloth"
uv pip install --python $PY unsloth unsloth_zoo

echo "### 4. verify + repair torch (unsloth may have downgraded it)"
$PY - <<'EOF'
import torch
ok = ("+cu128" in torch.__version__) and (torch.version.cuda == "12.8") and torch.cuda.is_available()
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
open("/tmp/torch_ok","w").write("1" if ok else "0")
EOF
if [ "$(cat /tmp/torch_ok)" != "1" ]; then
  echo ">>> torch was clobbered; reinstalling cu128 build"
  uv pip install --python $PY --reinstall torch --index-url $CU128
fi

echo "### 5. final import smoke check"
$PY - <<'EOF'
import torch
print("torch", torch.__version__, "sm_", "".join(map(str, torch.cuda.get_device_capability(0))))
import transformers, datasets, peft, trl, bitsandbytes
print("transformers", transformers.__version__, "| datasets", datasets.__version__,
      "| peft", peft.__version__, "| trl", trl.__version__, "| bnb", bitsandbytes.__version__)
import unsloth
print("unsloth import OK")
EOF
echo "### DONE"
