#pragma once

#ifdef _WIN32
#pragma warning(disable : 4365)
#pragma warning(disable : 4514)
#pragma warning(disable : 4710)
#pragma warning(disable : 4711)
#pragma warning(disable : 4820)
#pragma warning(disable : 4868)
#pragma warning(disable : 5246)
#pragma warning(disable : 6262)
#endif

#include "util/Log.hpp"

#include "util/py.hpp"

#include "TrainingArgs.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;

typedef int8_t int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;

template <typename T> inline size_t indexOf(const std::vector<T> &vec, const T &elem) {
    auto it = std::find(vec.begin(), vec.end(), elem);
    if (it == vec.end()) {
        throw std::invalid_argument("Element not found in vector");
    }
    return std::distance(vec.begin(), it);
}

template <typename T> inline bool contains(const std::vector<T> &vec, const T &elem) {
    return std::find(vec.begin(), vec.end(), elem) != vec.end();
}

template <typename T> inline void extend(std::vector<T> &vec, const std::vector<T> &other) {
    vec.insert(vec.end(), other.begin(), other.end());
}

template <typename T> inline void shuffle(std::vector<T> &vec) {
    std::mt19937 randomEngine(std::random_device{}());
    std::shuffle(vec.begin(), vec.end(), randomEngine);
}


#ifdef _WIN32
#pragma warning(push, 0)
#endif

#include <torch/script.h>
#include <torch/torch.h>

#ifdef _WIN32
#pragma warning(pop)
#endif


#include "position.h"
#include "bitboard.h"
#include "movegen.h"
#include "types.h"

using namespace Stockfish;

// Calculated as output of __precalculateMoveMappings() but defined here to be able to use it as a
// constexpr variable
static inline constexpr int ACTION_SIZE = 1814;

static inline constexpr int BOARD_LENGTH = 8;
static inline constexpr int BOARD_SIZE = BOARD_LENGTH * BOARD_LENGTH;

typedef std::pair<int, float> MoveScore;
