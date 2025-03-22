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

#include "Log.hpp"
#include "Time.hpp"
#include "chess.hpp"

#include "util/py.hpp"

#ifdef _WIN32
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#ifdef _WIN32
#pragma warning(pop)
#endif

using namespace chess;

static inline constexpr int ROW_COUNT = 8;
static inline constexpr int COLUMN_COUNT = 8;
// 6 types for each color
static inline constexpr int ENCODING_CHANNELS = 6 + 6 + 2;

// Calculated as output of __precalculateMoveMappings() but defined here to be able to use it as a
// constexpr variable
static inline constexpr int ACTION_SIZE = 1968;

template <typename T> inline size_t index_of(const std::vector<T> &vec, const T &elem) {
    auto it = std::find(vec.begin(), vec.end(), elem);
    if (it == vec.end()) {
        throw std::invalid_argument("Element not found in vector");
    }
    return std::distance(vec.begin(), it);
}

template <typename T> inline void extend(std::vector<T> &vec, const std::vector<T> &other) {
    vec.insert(vec.end(), other.begin(), other.end());
}

template <typename T> inline void shuffle(std::vector<T> &vec) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(vec.begin(), vec.end(), g);
}

std::string toString(const Move &move) { return move.uci(); }