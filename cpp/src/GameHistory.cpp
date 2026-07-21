#include "GameHistory.hpp"

#include "common.hpp"

Move findLegalMove(const Board &board, const std::string &moveUci) {
    const std::vector<Move> &legalMoves = board.validMoves();
    const auto matchingMove =
        std::find_if(legalMoves.begin(), legalMoves.end(),
                     [&moveUci](const Move move) { return toString(move) == moveUci; });
    if (matchingMove == legalMoves.end()) {
        throw std::invalid_argument("Illegal UCI move " + moveUci + " in position " + board.fen());
    }
    return *matchingMove;
}

Board replayMoves(const std::string &startingFen, const std::vector<std::string> &movesUci) {
    Board board(startingFen);
    for (const std::string &moveUci : movesUci) {
        board.makeMove(findLegalMove(board, moveUci));
    }
    return board;
}
