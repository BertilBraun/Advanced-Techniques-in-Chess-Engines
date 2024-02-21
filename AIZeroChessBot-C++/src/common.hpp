#pragma once

#include "chess.hpp"

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
#include <torch/torch.h>
#include <vector>

using namespace chess;

static inline constexpr int ROW_COUNT = 8;
static inline constexpr int COLUMN_COUNT = 8;
static inline constexpr int NUM_RES_BLOCKS = 8;
static inline constexpr int NUM_HIDDEN = 256;
// 6 types for each color
static inline constexpr int ENCODING_CHANNELS = 6 + 6;

// Calculated as output of __precalculateMoveMappings() but defined here to be able to use it as a
// constexpr variable
static inline constexpr int ACTION_SIZE = 1968;

static inline const std::string MEMORY_DIR_NAME = "memory";
static inline const std::string CONFIG_FILE_NAME = "last_training_config.pt";
static inline const std::filesystem::path COMMUNICATION_DIR = "communication";

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

std::vector<std::string> split(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

inline bool tqdm(size_t current, size_t total, std::string desc = "", int width = 50) {
    float progress = std::min((float) current / total, 1.0f);
    int pos = (int) (width * progress);

    std::cout << "[";
    for (int i = 0; i < width; ++i) {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " % " << desc << "\r";
    if (current == total) {
        std::cout << std::endl;
    }
    std::cout.flush();
    return current < total;
}

inline std::map<std::string, unsigned long long> __timeit_results;

constexpr bool TIME_CUDA_KERNELS = false;

// Time a function and add the result to the timeit results
// Should be callable like this:
// timeit([&] { return someFunction(); }, "someFunction");
template <typename Func> auto timeit(Func func, const std::string &funcName) {
    using ReturnType = decltype(func()); // Deduce the return type of the function

    if constexpr (TIME_CUDA_KERNELS) {
        if (torch::cuda::is_available()) {
            torch::cuda::synchronize();
        }
    }

    if constexpr (std::is_same_v<ReturnType, void>) {
        auto start = std::chrono::high_resolution_clock::now();

        // If the function returns void
        func(); // Just call the function

        if constexpr (TIME_CUDA_KERNELS) {
            if (torch::cuda::is_available()) {
                torch::cuda::synchronize();
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        __timeit_results[funcName] = duration;
    } else {
        auto start = std::chrono::high_resolution_clock::now();

        // If the function returns a value
        auto result = func(); // Call the function and store its result

        if constexpr (TIME_CUDA_KERNELS) {
            if (torch::cuda::is_available()) {
                torch::cuda::synchronize();
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        __timeit_results[funcName] = duration;

        return result; // Return the result of the function
    }
}

inline std::string get_timeit_results() {
    std::string result;
    for (auto &pair : __timeit_results) {
        result += pair.first + ": " + std::to_string(pair.second) + "ms\n";
    }
    return result;
}

inline std::string currentDateTime() {
    // Format: YYYY-MM-DD HH:MM:SS:MS
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
    return ss.str();
}