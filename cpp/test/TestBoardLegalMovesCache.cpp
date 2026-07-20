#include "position.h"

#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define private public
#include "Board.h"
#undef private

namespace {
void require(const bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void initializeStockfish() {
    Bitboards::init();
    Position::init();
}

void testRepeatedAccessReusesCache() {
    Board board;
    require(!board.m_validMoves.has_value(), "new board unexpectedly has cached moves");

    const std::vector<Move> &first = board.validMoves();
    require(board.m_validMoves.has_value(), "first access did not populate cache");
    const std::vector<Move> &second = board.validMoves();

    require(&first == &second, "repeated access returned a different vector");
    require(first.data() == second.data(), "repeated access replaced cached storage");
    require(first.size() == 20, "starting position did not have 20 legal moves");
}

void testMutationInvalidatesCache() {
    Board board;
    const Move move = board.validMoves().front();
    require(board.m_validMoves.has_value(), "move lookup did not populate cache");

    board.makeMove(move);
    require(!board.m_validMoves.has_value(), "makeMove did not invalidate cache");
    require(board.validMoves().size() == 20, "position after first move had unexpected move count");

    Board resetBoard;
    const std::vector<Move> movesBeforeReset = resetBoard.validMoves();
    resetBoard.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
    require(!resetBoard.m_validMoves.has_value(), "setFen did not invalidate cache");
    require(resetBoard.validMoves() != movesBeforeReset, "setFen returned stale cached moves");

    Board checkmateBoard("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
    require(checkmateBoard.isGameOver(), "checkmate position was not terminal; legal moves: " +
                                             std::to_string(checkmateBoard.validMoves().size()) +
                                             ", FEN: " + checkmateBoard.fen());
    require(checkmateBoard.m_validMoves.has_value(), "terminal detection did not populate cache");
    require(checkmateBoard.validMoves().empty(), "checkmate position had legal moves");
    require(checkmateBoard.checkWinner() == std::optional<int>(1),
            "checkmate position returned the wrong winner");
}

void testCopiesStartWithoutCache() {
    Board source;
    const std::vector<Move> &sourceMoves = source.validMoves();
    require(source.m_validMoves.has_value(), "source cache was not populated");

    Board copied(source);
    require(!copied.m_validMoves.has_value(), "copy constructor copied legal-move cache");
    const std::vector<Move> &copiedMoves = copied.validMoves();
    require(copiedMoves == sourceMoves, "copied board generated different moves");
    require(copiedMoves.data() != sourceMoves.data(), "copied board reused source cache storage");

    Board assigned;
    static_cast<void>(assigned.validMoves());
    assigned = source;
    require(!assigned.m_validMoves.has_value(), "copy assignment retained legal-move cache");
    require(assigned.validMoves() == sourceMoves, "assigned board generated different moves");
}
} // namespace

int main() {
    initializeStockfish();
    testRepeatedAccessReusesCache();
    testMutationInvalidatesCache();
    testCopiesStartWithoutCache();
    std::cout << "Board legal-move cache tests passed\n";
    return 0;
}
