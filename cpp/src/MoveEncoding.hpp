#pragma once

#include "common.hpp"

typedef std::array<std::array<std::array<int, PieceType::NUM_PIECE_TYPES>, BOARD_SIZE>, BOARD_SIZE>
    MoveMapping;

int encodeMove(const Move &move);

Move decodeMove(int moveIndex);

torch::Tensor encodeMoves(const std::vector<Move> &moves);

std::vector<Move> decodeMoves(const std::vector<int> &moveIndices);

std::vector<MoveScore> filterPolicyThenGetMovesAndProbabilities(const torch::Tensor &policy,
                                                                Board &board);

std::vector<MoveScore> filterMovesWithLegalMoves(const std::vector<MoveScore> &moves, Board &board);

std::vector<MoveScore>
filterPolicyWithEnPassantMovesThenGetMovesAndProbabilities(const torch::Tensor &policy,
                                                           Board &board);
