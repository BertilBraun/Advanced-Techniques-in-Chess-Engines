#pragma once

#include "common.hpp"

// 6 types for each color + 4 for castling rights + 1 for en passant
static constexpr int ENCODING_CHANNELS = 6 + 6 + 4 + 1;

typedef std::array<std::array<std::array<int8, BOARD_LENGTH>, BOARD_LENGTH>, ENCODING_CHANNELS>
    EncodedBoard;
typedef std::array<uint64, ENCODING_CHANNELS> CompressedEncodedBoard;

// A decent 64-bit hash for CompressedEncdodedBoard.
static constexpr uint64 splitmix64(uint64 x) noexcept {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

struct BoardHash {
    size_t operator()(const CompressedEncodedBoard &b) const noexcept {
        uint64 h = 0xcbf29ce484222325ull; // seed
        for (uint64 v : b) {              // mix each channel
            h ^= splitmix64(v);
            h = splitmix64(h);
        }
        return static_cast<size_t>(h); // must return size_t
    }
};

CompressedEncodedBoard encodeBoard(const Board *board);

torch::Tensor toTensor(const CompressedEncodedBoard &compressed, torch::Device device);

std::optional<float> getBoardResultScore(const Board &board);
