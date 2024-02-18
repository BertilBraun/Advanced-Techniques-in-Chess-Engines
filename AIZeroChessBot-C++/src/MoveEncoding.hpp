#pragma once

#include "common.hpp"

std::vector<std::pair<Move, float>>
filterPolicyThenGetMovesAndProbabilities(const torch::Tensor &policy, Board &board);

int encodeMove(const Move &move);

Move decodeMove(int moveIndex);

std::vector<Move> decodeMoves(const std::vector<int> &moveIndices);

Square flipSquareHorizontal(const Square &square);

Square flipSquareVertical(const Square &square);

torch::Tensor flipActionProbabilities(const torch::Tensor &actionProbabilities,
                                      const std::function<int(int)> &flipMoveIndex);

int flipMoveIndexHorizontal(int moveIndex);

int flipMoveIndexVertical(int moveIndex);
