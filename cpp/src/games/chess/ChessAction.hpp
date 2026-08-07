#pragma once

#include "types.h"

#include <string>

class Board;

struct ChessAction {
    static constexpr int action_count = 1880;

    Stockfish::Move move;

    explicit ChessAction(const Stockfish::Move chessMove) : move(chessMove) {}
    [[nodiscard]] int encode(const Board &position) const;
    [[nodiscard]] static ChessAction decode(int actionId, const Board &position);
    [[nodiscard]] std::string toUci() const;
    [[nodiscard]] bool operator==(const ChessAction &) const noexcept = default;
};
