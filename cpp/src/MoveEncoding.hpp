#pragma once

#include "common.hpp"

[[nodiscard]] int encodeMove(Move move, const Board *board);

[[nodiscard]] std::vector<Move> decodeMoves(const std::vector<int> &moveIndices,
                                            const Board *board);

[[nodiscard]] std::vector<EncodedMoveScore>
filterPolicyThenGetMovesAndProbabilities(const torch::Tensor &policy, const Board *board);
