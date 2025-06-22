
#include "chess.hpp"
#include <chrono>
#include <iostream>

using namespace chess;

void generateMovesBenchmark(Board &board, int depth = 3) {
    if (depth < 1) {
        return;
    }
    for (Move move : board.legalMoves()) {
        Board boardCopy = board.copy();
        boardCopy.push(move);
        generateMovesBenchmark(boardCopy, depth - 1);
    }
}

void generateMovesBenchmark(int iterations, int depth) {
    auto totalDuration = std::chrono::duration<double>::zero();

    for (int i = 0; i < iterations; ++i) {
        Board board;
        auto start = std::chrono::high_resolution_clock::now();

        generateMovesBenchmark(board, depth);

        auto end = std::chrono::high_resolution_clock::now();
        totalDuration += end - start;
    }

    double averageTime = totalDuration.count() / iterations;
    std::cout << "Average time for generating moves and pushing/popping a move: " << averageTime
              << " seconds.\n";
    std::cout << "Total time for generating moves and pushing/popping a move: "
              << totalDuration.count() << " seconds.\n";
}

void copyStateBenchmark(int iterations) {
    Board board; // Initialize your board
    auto totalDuration = std::chrono::duration<double>::zero();

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        Board boardCopy = board.copy();

        auto end = std::chrono::high_resolution_clock::now();
        totalDuration += end - start;
    }

    double averageTime = totalDuration.count() / iterations;
    std::cout << "Average time for copying board state: " << averageTime << " seconds.\n";
    std::cout << "Total time for copying board state: " << totalDuration.count() << " seconds.\n";
}

int main() {
    generateMovesBenchmark(100, 4);
    copyStateBenchmark(100'000'000);
    return 0;
}
