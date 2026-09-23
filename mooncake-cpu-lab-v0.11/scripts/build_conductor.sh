#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
mc_root=vendor/Mooncake
"${CXX:-g++}" -std=c++20 -O2 -fPIC -shared \
  -I native/logshim -I "$mc_root/mooncake-conductor/include" \
  -I "$mc_root/mooncake-common/include" \
  native/conductor_bridge.cpp \
  "$mc_root/mooncake-conductor/src/prefixindex/prefix_indexer.cpp" \
  "$mc_root/mooncake-conductor/src/prefixindex/hash_strategy.cpp" \
  -lcrypto -pthread -o build/libconductor_lab.so
