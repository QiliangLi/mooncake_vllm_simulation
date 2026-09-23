#pragma once
// Standalone lab only: the indexer uses LOG(ERROR) for three diagnostic paths.
// Scheduling, hashing, cache ownership and query code are upstream originals.
#include <iostream>
#define LOG(severity) std::cerr
#define LOG_EVERY_N(severity, n) std::cerr
