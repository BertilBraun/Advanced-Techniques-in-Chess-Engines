#include "games/chess/implementation/ChessAction.hpp"

std::string ChessAction::toUci() const {
    if (move == Stockfish::Move::null()) {
        return "null";
    }
    const auto appendSquare = [](std::string &uci, const Stockfish::Square square) {
        uci += static_cast<char>('a' + Stockfish::file_of(square));
        uci += static_cast<char>('1' + Stockfish::rank_of(square));
    };
    const Stockfish::Square from = move.from_sq();
    const Stockfish::Square to = move.type_of() == Stockfish::CASTLING
                                     ? static_cast<Stockfish::Square>(
                                           static_cast<int>(from) + (move.to_sq() > from ? 2 : -2))
                                     : move.to_sq();
    std::string uci;
    uci.reserve(5);
    appendSquare(uci, from);
    appendSquare(uci, to);
    if (move.type_of() == Stockfish::PROMOTION) {
        uci += "  nbrqk"[move.promotion_type()];
    }
    return uci;
}
