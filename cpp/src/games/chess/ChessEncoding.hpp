#pragma once

#include "BoardEncoding.hpp"

struct ChessRepresentationDimensions {
    static constexpr int board_length = BOARD_LEN;
    static constexpr int channel_count = BOARD_C;
    static constexpr int binary_channel_count = BINARY_C;
    static constexpr int scalar_channel_count = SCALAR_C;
    static constexpr int action_count = ACTION_SIZE;
};

using EncodedChessPosition = CompressedEncodedBoard;

[[nodiscard]] inline EncodedChessPosition encode_chess_position(const Board &position) {
    return encodeBoard(&position);
}

inline void encode_chess_position_into(const Board &position, int8 *destination) {
    encodeBoardInto(position, destination);
}
