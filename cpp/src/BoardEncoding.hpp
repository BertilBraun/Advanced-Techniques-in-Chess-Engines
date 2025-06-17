#pragma once

#include "common.hpp"

// ------------ board layout --------------------------------------------------
// 12 piece-type bit-boards  (6 white + 6 black)
// 4  castling-rights planes
// 2  occupancy planes       (all white pieces, all black pieces)
// 1  "checkers" plane
// 6  material-difference scalar planes
static constexpr int BOARD_LEN = 8;
static constexpr int BOARD_C = 25;  // total channels
static constexpr int BINARY_C = 19; // stored as one u64 each
static constexpr int SCALAR_C = 6;  // stored as one  int8 each
static_assert(BOARD_C == BINARY_C + SCALAR_C);

// ---------------------------------------------------------------------------
// in-memory compressed format
// ---------------------------------------------------------------------------
struct CompressedEncodedBoard {
    std::array<uint64, BINARY_C> bits; // 19 × 8  B
    std::array<int8, SCALAR_C> scal;   //  6 × 1  B

    [[nodiscard]] bool operator==(const CompressedEncodedBoard &other) const noexcept {
        return bits == other.bits && scal == other.scal;
    }
};

struct BoardHash {
    [[nodiscard]] std::size_t operator()(CompressedEncodedBoard const &b) const noexcept;
};

[[nodiscard]] CompressedEncodedBoard encodeBoard(const Board *board);

[[nodiscard]] torch::Tensor toTensor(const CompressedEncodedBoard &compressed);

[[nodiscard]] float getBoardResultScore(const Board &board);
