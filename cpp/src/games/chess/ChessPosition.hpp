#pragma once

#include "games/chess/ChessBoard.hpp"
#include "games/chess/ChessHistory.hpp"

using ChessPosition = Board;

[[nodiscard]] inline ChessPosition
replay_chess_position(const std::string &starting_fen, const std::vector<std::string> &moves_uci) {
    return replayMoves(starting_fen, moves_uci);
}
