#pragma once

#include "common.hpp"

int encodeMove(Move move, const Board& board);

torch::Tensor encodeMoves(const std::vector<Move> &moves, const Board& board);

std::vector<Move> decodeMoves(const std::vector<int> &moveIndices, const Board& board);

std::vector<MoveScore> filterPolicyThenGetMovesAndProbabilities(const torch::Tensor &policy, const Board &board);


