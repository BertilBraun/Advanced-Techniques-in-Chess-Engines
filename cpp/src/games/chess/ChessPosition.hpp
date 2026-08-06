#pragma once

#include "Board.h"
#include "GameHistory.hpp"

using ChessPosition = Board;

[[nodiscard]] inline ChessPosition
replay_chess_position(const std::string &starting_fen, const std::vector<std::string> &moves_uci) {
    return replayMoves(starting_fen, moves_uci);
}
