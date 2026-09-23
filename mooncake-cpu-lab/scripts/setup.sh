#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo 'Install uv first: python3 -m pip install uv'; exit 1; }
command -v g++ >/dev/null || { echo 'Install a C++20 compiler and libssl-dev first.'; exit 1; }
uv venv --python 3.12 --allow-existing .venv
if [ "$(uname)" = "Darwin" ]; then
  # The +cpu index has no Darwin wheels and vLLM publishes no macOS wheel;
  # plain torch from PyPI matches vLLM's own requirements/cpu.txt Darwin line.
  uv pip install --python .venv/bin/python 'torch==2.11.0'
  uv pip install --python .venv/bin/python -r requirements.lock
  uv pip install --python .venv/bin/python --no-deps 'xgrammar==0.1.32'
  # Metadata-only stub for importlib.metadata.version('vllm') in run.py;
  # the executed code is the hash-audited vendor/vllm source tree.
  sp=.venv/lib/python3.12/site-packages/vllm-0.20.2.dist-info
  mkdir -p "$sp"
  printf 'Metadata-Version: 2.1\nName: vllm\nVersion: 0.20.2\n' > "$sp/METADATA"
  printf 'vllm-0.20.2.dist-info/METADATA,,\nvllm-0.20.2.dist-info/RECORD,,\n' > "$sp/RECORD"
  # Apple's CLT may lack C++ headers; prefer Homebrew gcc when present.
  if [ -z "${CXX:-}" ] && command -v brew >/dev/null; then
    bgxx="$(ls "$(brew --prefix gcc)"/bin/g++-* 2>/dev/null | sort -V | tail -1 || true)"
    [ -n "$bgxx" ] && export CXX="$bgxx"
  fi
else
  uv pip install --offline --python .venv/bin/python 'torch==2.11.0+cpu' --index-url https://download.pytorch.org/whl/cpu || \
    uv pip install --python .venv/bin/python 'torch==2.11.0+cpu' --index-url https://download.pytorch.org/whl/cpu
  uv pip install --offline --python .venv/bin/python -r requirements.lock || \
    uv pip install --python .venv/bin/python -r requirements.lock
  # The device worker and GPU extension are never imported/executed. Avoid installing CUDA dependencies.
  uv pip install --offline --python .venv/bin/python --no-deps 'vllm==0.20.2' 'xgrammar==0.1.32' || \
    uv pip install --python .venv/bin/python --no-deps 'vllm==0.20.2' 'xgrammar==0.1.32'
fi
bash scripts/build_conductor.sh
.venv/bin/python -m pytest -q tests
