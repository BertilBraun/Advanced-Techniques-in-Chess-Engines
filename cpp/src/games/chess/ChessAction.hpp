#pragma once

#include "MoveEncoding.hpp"

using ChessAction = Move;

[[nodiscard]] inline int chess_action_id(const ChessAction action, const Board &position) {
    return encodeMove(action, &position);
}

[[nodiscard]] inline std::vector<ChessAction>
decode_chess_actions(const std::vector<int> &action_ids, const Board &position) {
    return decodeMoves(action_ids, &position);
}
