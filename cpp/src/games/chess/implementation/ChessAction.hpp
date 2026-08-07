#pragma once

#include "types.h"

#include <string>

class Board;

struct ChessAction {
    Stockfish::Move move;

    explicit ChessAction(const Stockfish::Move chessMove) : move(chessMove) {}
    [[nodiscard]] std::string toUci() const;
    [[nodiscard]] bool operator==(const ChessAction &) const noexcept = default;
};
