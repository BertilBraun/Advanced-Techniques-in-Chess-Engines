#pragma once

#include "types.h"

#include <string>
#include <vector>

class Board;

struct ChessAction {
    static constexpr int action_count = 1880;

    Stockfish::Move move;

    explicit ChessAction(const Stockfish::Move chessMove) : move(chessMove) {}
    [[nodiscard]] bool operator==(const ChessAction &) const noexcept = default;
};

class ChessActionCodec {
public:
    [[nodiscard]] static int encode(ChessAction action, const Board &position);
    [[nodiscard]] static std::vector<ChessAction> decode(const std::vector<int> &actionIds,
                                                         const Board &position);
    [[nodiscard]] static std::string toUci(const ChessAction action) {
        if (action.move == Stockfish::Move::null()) {
            return "null";
        }
        const auto appendSquare = [](std::string &uci, const Stockfish::Square square) {
            uci += static_cast<char>('a' + Stockfish::file_of(square));
            uci += static_cast<char>('1' + Stockfish::rank_of(square));
        };
        const Stockfish::Square from = action.move.from_sq();
        const Stockfish::Square to =
            action.move.type_of() == Stockfish::CASTLING
                ? static_cast<Stockfish::Square>(static_cast<int>(from) +
                                                 (action.move.to_sq() > from ? 2 : -2))
                : action.move.to_sq();
        std::string uci;
        uci.reserve(5);
        appendSquare(uci, from);
        appendSquare(uci, to);
        if (action.move.type_of() == Stockfish::PROMOTION) {
            uci += "  nbrqk"[action.move.promotion_type()];
        }
        return uci;
    }
};
