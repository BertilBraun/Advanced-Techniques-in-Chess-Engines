#pragma once

#include "games/chess/ChessBoard.hpp"

#include <vector>

struct ChessAction {
    Move move;

    explicit ChessAction(const Move chessMove) : move(chessMove) {}
    [[nodiscard]] bool operator==(const ChessAction &) const noexcept = default;
};

class ChessActionCodec {
public:
    [[nodiscard]] static int encode(ChessAction action, const Board &position);
    [[nodiscard]] static std::vector<ChessAction> decode(const std::vector<int> &actionIds,
                                                         const Board &position);
};
