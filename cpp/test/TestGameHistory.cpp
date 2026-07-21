#include "position.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "GameHistory.hpp"

namespace {
void require(const bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireInvalidHistory(const std::vector<std::string> &moves) {
    try {
        static_cast<void>(replayMoves(Board{}.fen(), moves));
    } catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error("invalid history was accepted");
}

void testReplayPreservesRepetition() {
    const std::vector<std::string> moves{
        "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
    };
    const Board board = replayMoves(Board{}.fen(), moves);
    require(board.repetitionCount() == 2, "replay did not preserve two earlier occurrences");
    require(board.isGameOver(), "third occurrence was not terminal");
}

void testReplayRejectsInvalidMoves() {
    requireInvalidHistory({"e2e5"});
}
} // namespace

int main() {
    Bitboards::init();
    Position::init();
    testReplayPreservesRepetition();
    testReplayRejectsInvalidMoves();
    std::cout << "Game history tests passed\n";
    return 0;
}
