#include "TestRunner.hpp"
#include "position.h"

#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "games/chess/implementation/ChessBoard.hpp"

struct BoardTestAccess {
    [[nodiscard]] static bool hasCachedMoves(const Board &board) { return board.m_validMoves.has_value(); }
};

namespace {
void require(const bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void initializeStockfish() {
    Stockfish::Bitboards::init();
    Stockfish::Position::init();
}

void testRepeatedAccessReusesCache() {
    Board board;
    require(!BoardTestAccess::hasCachedMoves(board), "new board unexpectedly has cached moves");

    const std::vector<Stockfish::Move> &first = board.validMoves();
    require(BoardTestAccess::hasCachedMoves(board), "first access did not populate cache");
    const std::vector<Stockfish::Move> &second = board.validMoves();

    require(&first == &second, "repeated access returned a different vector");
    require(first.data() == second.data(), "repeated access replaced cached storage");
    require(first.size() == 20, "starting position did not have 20 legal moves");
}

void testMutationInvalidatesCache() {
    Board board;
    const Stockfish::Move move = board.validMoves().front();
    require(BoardTestAccess::hasCachedMoves(board), "move lookup did not populate cache");

    board.makeMove(move);
    require(!BoardTestAccess::hasCachedMoves(board), "makeMove did not invalidate cache");
    require(board.validMoves().size() == 20, "position after first move had unexpected move count");

    Board resetBoard;
    const std::vector<Stockfish::Move> movesBeforeReset = resetBoard.validMoves();
    resetBoard.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
    require(!BoardTestAccess::hasCachedMoves(resetBoard), "setFen did not invalidate cache");
    require(resetBoard.validMoves() != movesBeforeReset, "setFen returned stale cached moves");

    Board checkmateBoard("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
    require(checkmateBoard.isGameOver(), "checkmate position was not terminal; legal moves: " +
                                             std::to_string(checkmateBoard.validMoves().size()) +
                                             ", FEN: " + checkmateBoard.fen());
    require(BoardTestAccess::hasCachedMoves(checkmateBoard), "terminal detection did not populate cache");
    require(checkmateBoard.validMoves().empty(), "checkmate position had legal moves");
    require(checkmateBoard.checkWinner() == std::optional<int>(1),
            "checkmate position returned the wrong winner");
}

void testCopiesStartWithoutCache() {
    Board source;
    const std::vector<Stockfish::Move> &sourceMoves = source.validMoves();
    require(BoardTestAccess::hasCachedMoves(source), "source cache was not populated");

    Board copied(source);
    require(!BoardTestAccess::hasCachedMoves(copied), "copy constructor copied legal-move cache");
    const std::vector<Stockfish::Move> &copiedMoves = copied.validMoves();
    require(copiedMoves == sourceMoves, "copied board generated different moves");
    require(copiedMoves.data() != sourceMoves.data(), "copied board reused source cache storage");

    Board assigned;
    static_cast<void>(assigned.validMoves());
    assigned = source;
    require(!BoardTestAccess::hasCachedMoves(assigned), "copy assignment retained legal-move cache");
    require(assigned.validMoves() == sourceMoves, "assigned board generated different moves");
}

void testKnightOnlyInsufficientMaterial() {
    const Board knightVersusKing("8/8/8/4kn2/8/8/6K1/8 w - - 2 78");
    require(knightVersusKing.isGameOver(), "king and knight versus king was not terminal");

    const Board twoKnightsVersusKing("8/8/8/4knn1/8/8/6K1/8 w - - 2 78");
    require(!twoKnightsVersusKing.isGameOver(), "king and two knights versus king was terminal");
}
} // namespace

int runBoardLegalMovesCacheTests() {
    initializeStockfish();
    testRepeatedAccessReusesCache();
    testMutationInvalidatesCache();
    testCopiesStartWithoutCache();
    testKnightOnlyInsufficientMaterial();
    std::cout << "Board legal-move cache tests passed\n";
    return 0;
}
