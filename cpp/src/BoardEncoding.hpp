#pragma once

#include "common.hpp"

// 6 types for each color + 4 for castling rights + 1 for en passant
static constexpr int ENCODING_CHANNELS = 6 + 6 + 4 + 1;

typedef std::array<std::array<std::array<int8, BOARD_LENGTH>, BOARD_LENGTH>, ENCODING_CHANNELS>
    EncodedBoard;
typedef std::array<uint64, ENCODING_CHANNELS> CompressedEncodedBoard;

CompressedEncodedBoard encodeBoard(const Board *board);

torch::Tensor toTensor(const CompressedEncodedBoard &compressed, torch::Device device);

uint64 hash(const CompressedEncodedBoard &compressed);

std::optional<float> getBoardResultScore(const Board &board);
