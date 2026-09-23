#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo 'Install uv first: python3 -m pip install uv'; exit 1; }
command -v g++ >/dev/null || { echo 'Install a C++20 compiler and libssl-dev first.'; exit 1; }
uv venv --python 3.12 --allow-existing .venv
uv pip install --offline --python .venv/bin/python 'torch==2.11.0+cpu' --index-url https://download.pytorch.org/whl/cpu || \
  uv pip install --python .venv/bin/python 'torch==2.11.0+cpu' --index-url https://download.pytorch.org/whl/cpu
uv pip install --offline --python .venv/bin/python -r requirements.lock || \
  uv pip install --python .venv/bin/python -r requirements.lock
# The device worker and GPU extension are never imported/executed. Avoid installing CUDA dependencies.
uv pip install --offline --python .venv/bin/python --no-deps 'vllm==0.20.2' 'xgrammar==0.1.32' || \
  uv pip install --python .venv/bin/python --no-deps 'vllm==0.20.2' 'xgrammar==0.1.32'
bash scripts/build_conductor.sh
.venv/bin/python -m pytest -q tests
