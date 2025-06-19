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

#include "util/TimeItGuard.h"

#include "util/py.hpp"

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
#include <new> // for placement‐new
#include <optional>
#include <queue>
#include <random>
#include <ranges>
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

template <typename T> inline size_t argmax(const std::vector<T> &vec) {
    if (vec.empty()) {
        throw std::invalid_argument("Cannot find argmax of an empty vector");
    }
    return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
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

// Calculated as output of __precalculateMoveMappings() but defined here to be able to use it as a
// constexpr variable
static inline constexpr int ACTION_SIZE = 1814;

static inline constexpr int BOARD_LENGTH = 8;
static inline constexpr int BOARD_SIZE = BOARD_LENGTH * BOARD_LENGTH;

#include "types.h"

using namespace Stockfish;

inline std::pair<int, int> squareToIndex(const int square) {
    return {square / BOARD_LENGTH, square % BOARD_LENGTH};
}

inline Square square(const int col, const int row) {
    // Converts a column and row to a square index.
    // param col: The column index (0-7).
    // param row: The row index (0-7).
    // :return: The square index.
    assert(col >= 0 && col < BOARD_LENGTH && "Column index out of bounds");
    assert(row >= 0 && row < BOARD_LENGTH && "Row index out of bounds");
    return static_cast<Square>(row * BOARD_LENGTH + col);
}

static inline constexpr std::array COLORS = {Color::WHITE, Color::BLACK};
static inline constexpr std::array PIECE_TYPES = {PieceType::PAWN,   PieceType::KNIGHT,
                                                  PieceType::BISHOP, PieceType::ROOK,
                                                  PieceType::QUEEN,  PieceType::KING};

#include "Board.h"

#ifdef _WIN32
#pragma warning(push, 0)
#endif

#include <torch/script.h>
#include <torch/torch.h>

#ifdef _WIN32
#pragma warning(pop)
#endif

typedef std::pair<Move, float> MoveScore;
typedef std::pair<int, float> EncodedMoveScore;

inline std::string toString(const Square square) {
    File file = file_of(square);
    Rank rank = rank_of(square);

    std::stringstream ss;
    ss << static_cast<char>('a' + file) << (rank + 1);
    return ss.str();
}

inline std::string toString(const Move move) {
    // Converts a Move to a string representation.
    // Since Stockfish does not provide a direct way to convert Move to string,
    // we use the UCI format.
    if (move == Move::null()) {
        return "null";
    }

    std::stringstream ss;
    ss << toString(move.from_sq()) << toString(move.to_sq());
    if (move.type_of() == PROMOTION) {
        ss << " pkbrqK"[move.promotion_type()];
    }
    return ss.str();
}

