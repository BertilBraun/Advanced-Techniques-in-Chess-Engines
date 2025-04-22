#pragma once

#include "common.hpp"

#include "MoveEncoding.cpp"

void testMoveMappingCount() {
    auto [moveMappings, index] = precalculateMoveMappings();
    assert(index == ACTION_SIZE);
}

void fuzzTestMoveEncodingAndDecoding() {
    for (auto i : range(100)) {
        tqdm(i, 100, "Fuzz testing move encoding and decoding");
        Board board;
        while (!board.isGameOver()) {
            for (const Move &move : board.legalMoves()) {
                assert(decodeMove(encodeMove(move)) == move);
            }
            board.push(board.legalMoves()[rand() % board.legalMoves().size()]);
        }
    }
}