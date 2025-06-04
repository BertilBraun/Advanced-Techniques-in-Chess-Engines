#pragma once

#include "common.hpp"

int encodeMove(const Move &move);

Move decodeMove(int moveIndex);

torch::Tensor encodeMoves(const std::vector<Move> &moves);

std::vector<Move> decodeMoves(const std::vector<int> &moveIndices);

std::vector<MoveScore> filterPolicyThenGetMovesAndProbabilities(const torch::Tensor &policy, Board &board);


