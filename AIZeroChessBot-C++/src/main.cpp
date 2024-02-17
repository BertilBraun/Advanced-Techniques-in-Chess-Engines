#include "chess.hpp"
#include <iostream>


using namespace chess;

int main() {

    std::cout << "Hello, World!" << std::endl;

    Board board;
    std::cout << board.boardFen() << std::endl;
    board.push(Move::fromUci("e2e4"));
    std::cout << board.boardFen() << std::endl;

    for (auto move : board.legalMoves()) {
        std::cout << move.uci() << std::endl;
    }

    board.push(board.legalMoves()[0]);

    return 0;
}