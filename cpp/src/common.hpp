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

#include "chess.hpp"
#include "util/Log.hpp"
#include "util/Time.hpp"

#include "util/py.hpp"

#include "TrainingArgs.hpp"

#ifdef _WIN32
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
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

#include <torch/script.h>
#include <torch/torch.h>

#include <tensorboard_logger.h>

#include <fmt/core.h>

#ifdef _WIN32
#pragma warning(pop)
#endif

using namespace chess;

typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;

typedef int8_t int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;

// Calculated as output of __precalculateMoveMappings() but defined here to be able to use it as a
// constexpr variable
static inline constexpr int ACTION_SIZE = 1968;

static inline constexpr int BOARD_LENGTH = 8;
static inline constexpr int BOARD_SIZE = BOARD_LENGTH * BOARD_LENGTH;

typedef std::pair<int, float> MoveScore;

thread_local std::mt19937 _random_engine(std::random_device{}());

template <typename T> inline size_t indexOf(const std::vector<T> &vec, const T &elem) {
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
    std::shuffle(vec.begin(), vec.end(), _random_engine);
}

std::string toString(const Move &move) { return move.uci(); }

inline std::pair<int, int> squareToIndex(int square) {
    return {square / BOARD_LENGTH, square % BOARD_LENGTH};
}

std::vector<float> dirichlet(float alpha, size_t n) {
    // Sample from a Dirichlet distribution with parameter alpha.
    std::gamma_distribution<float> gamma(alpha, 1.0);

    std::vector<float> noise(n);
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        noise[i] = gamma(_random_engine);
        sum += noise[i];
    }

    float norm = 1.0f / sum;
    for (size_t i = 0; i < n; i++) {
        noise[i] *= norm;
    }

    return noise;
}

// It returns a pair: (latest model file path, iteration number).
std::pair<std::string, int> get_latest_iteration_save_path(const std::string &savePath) {
    // Models are saved in the savePath folder numbered starting from 1 up to the latest iteration.
    // For example: "{savePath}/model_1.pt", "{savePath}/model_2.pt", etc.
    // The function returns the latest model file path and its iteration number.

    for (int i : range(500, 0, -1)) {
        const std::string modelPath = savePath + "/model_" + std::to_string(i) + ".jit.pt";
        if (std::filesystem::exists(std::filesystem::path(modelPath))) {
            return {modelPath, i};
        }
    }
    assert(false && "No model found in the save path.");
}
