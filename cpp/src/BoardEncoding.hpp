#pragma once

#include "common.hpp"
#include "util/BitBoard.hpp"
#include "util/PackedPlane.hpp"

// ------------ board layout --------------------------------------------------
// 12 piece-type bit-boards  (6 white + 6 black)
// 4  castling-rights planes
// 2  occupancy planes       (all white pieces, all black pieces)
// 1  "checkers" plane
// 1  en-passant target plane
// 2  repetition planes      (second and third occurrence)
// 6  material-difference scalar planes
// 1  halfmove-clock scalar plane
static constexpr int BOARD_LEN = 8;
static constexpr int BOARD_C = 29;
static constexpr int BINARY_C = 22;
static constexpr int SCALAR_C = 7;
static_assert(BOARD_C == BINARY_C + SCALAR_C);

// ---------------------------------------------------------------------------
// in-memory compressed format
// ---------------------------------------------------------------------------
struct CompressedEncodedBoard {
    std::array<BitBoard<BOARD_LEN>, BINARY_C> bits;
    std::array<int8, SCALAR_C> scal;

    [[nodiscard]] bool operator==(const CompressedEncodedBoard &other) const noexcept {
        return bits == other.bits && scal == other.scal;
    }
};

inline constexpr std::size_t CHESS_PACKED_BINARY_BYTES = packed_binary_plane_bytes<BOARD_LEN, BINARY_C>;
inline constexpr std::size_t CHESS_PACKED_BYTES =
    CHESS_PACKED_BINARY_BYTES + static_cast<std::size_t>(SCALAR_C) * sizeof(int8);

struct BoardFingerprint {
    uint64 first;
    uint64 second;

    [[nodiscard]] bool operator==(const BoardFingerprint &other) const noexcept {
        return first == other.first && second == other.second;
    }
};

struct BoardFingerprintHash {
    [[nodiscard]] std::size_t operator()(const BoardFingerprint &fingerprint) const noexcept;
};

[[nodiscard]] CompressedEncodedBoard encodeBoard(const Board *board);

[[nodiscard]] BoardFingerprint fingerprintBoard(const CompressedEncodedBoard &compressed);

[[nodiscard]] torch::Tensor toTensor(const CompressedEncodedBoard &compressed);

void writeTensorEncoding(const CompressedEncodedBoard &compressed, int8 *destination);

void writePackedPlaneEncoding(const CompressedEncodedBoard &compressed, int8 *destination);

void encodeBoardInto(const Board &board, int8 *destination);

[[nodiscard]] float getBoardResultScore(const Board &board);
