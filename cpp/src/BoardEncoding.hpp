#pragma once

#include "common.hpp"

// 6 types for each color + 1 for castling and en passant + 1 for current player
static constexpr int ENCODING_CHANNELS = 6 + 6 + 2;

typedef std::array<std::array<std::array<int8, BOARD_LENGTH>, BOARD_LENGTH>, ENCODING_CHANNELS>
    EncodedBoard;
typedef std::array<uint64, ENCODING_CHANNELS> CompressedEncodedBoard;

CompressedEncodedBoard encodeBoard(const Board &board);

CompressedEncodedBoard compress(const EncodedBoard &binary);

EncodedBoard decompress(const CompressedEncodedBoard &compressed);

torch::Tensor toTensor(const CompressedEncodedBoard &compressed, torch::Device device);

uint64 hash(const CompressedEncodedBoard &compressed);

Board decodeBoard(const EncodedBoard &encodedBoard);

float getMaterialScore(const Board &board);

std::optional<float> getBoardResultScore(Board &board);
