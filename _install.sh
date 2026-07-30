#!/usr/bin/env bash
# Unattended setup: Miniconda + conda env `voice` + torch(cu124) + NeMo/Kokoro stack.
# Auto-falls back to python 3.10 if nemo_toolkit[asr] won't resolve on 3.11.
set -eo pipefail
export HOME=/home/omras
cd "$HOME"

echo "=== [1/4] Miniconda ==="
if [ ! -d "$HOME/miniconda3" ]; then
  URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  if command -v wget >/dev/null; then wget -q "$URL" -O /tmp/mc.sh
  else curl -fsSL "$URL" -o /tmp/mc.sh; fi
  bash /tmp/mc.sh -b -p "$HOME/miniconda3"
else
  echo "miniconda already present"
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# conda 25.x requires accepting channel ToS before non-interactive env create.
# `|| true` so older conda (no `tos` subcommand) still proceeds.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

install_stack() {
  conda run -n voice pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 || return 1
  conda run -n voice pip install "nemo_toolkit[asr]" "kokoro>=0.9.4" soundfile transformers accelerate bitsandbytes || return 1
}

echo "=== [2/4] conda env (python 3.11) ==="
conda env remove -n voice -y >/dev/null 2>&1 || true
conda create -n voice python=3.11 -y

echo "=== [3/4] pip stack (py3.11) ==="
if install_stack; then
  echo "=== stack OK on python 3.11 ==="
else
  echo "=== py3.11 stack FAILED -> falling back to python 3.10 ==="
  conda env remove -n voice -y
  conda create -n voice python=3.10 -y
  echo "=== [3b/4] pip stack (py3.10) ==="
  install_stack
  echo "=== stack OK on python 3.10 ==="
fi

echo "=== [4/4] python version in env ==="
conda run -n voice python --version
echo "INSTALL_DONE"
