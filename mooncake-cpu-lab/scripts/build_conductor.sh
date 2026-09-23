#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
mc_root=vendor/Mooncake
extra=()
if [ "$(uname)" = "Darwin" ]; then
  # Apple SDKs ship no OpenSSL headers; the vendored hash code includes <openssl/*.h>.
  ssl="$(brew --prefix openssl@3 2>/dev/null || true)"
  if [ -n "$ssl" ]; then extra+=(-I "$ssl/include" -L "$ssl/lib"); fi
fi
"${CXX:-g++}" -std=c++20 -O2 -fPIC -shared \
  -I native/logshim -I "$mc_root/mooncake-conductor/include" \
  -I "$mc_root/mooncake-common/include" \
  ${extra[@]+"${extra[@]}"} \
  native/conductor_bridge.cpp \
  "$mc_root/mooncake-conductor/src/prefixindex/prefix_indexer.cpp" \
  "$mc_root/mooncake-conductor/src/prefixindex/hash_strategy.cpp" \
  -lcrypto -pthread -o build/libconductor_lab.so
