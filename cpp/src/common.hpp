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

#include "commonBase.hpp"

#ifdef _WIN32
#pragma warning(push, 0)
#endif

#include <torch/script.h>
#include <torch/torch.h>

#include "util/TensorBoardLogger.hpp"

#ifdef _WIN32
#pragma warning(pop)
#endif

using namespace chess;

// Calculated as output of __precalculateMoveMappings() but defined here to be able to use it as a
// constexpr variable
static inline constexpr int ACTION_SIZE = 1968;

static inline constexpr int BOARD_LENGTH = 8;
static inline constexpr int BOARD_SIZE = BOARD_LENGTH * BOARD_LENGTH;

typedef std::pair<int, float> MoveScore;

inline std::string toString(const Move &move) { return move.uci(); }

inline std::pair<int, int> squareToIndex(int square) {
    return {square / BOARD_LENGTH, square % BOARD_LENGTH};
}

std::vector<float> dirichlet(float alpha, size_t n);

// It returns a pair: (latest model file path, iteration number).
std::pair<std::string, int> get_latest_iteration_save_path(const std::string &savePath);

size_t current_time_step();