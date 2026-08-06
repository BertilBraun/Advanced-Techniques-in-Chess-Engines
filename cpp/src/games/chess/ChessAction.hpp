#pragma once

#include "common.hpp"

[[nodiscard]] int encodeMove(Move move, const Board *board);

[[nodiscard]] std::vector<Move> decodeMoves(const std::vector<int> &moveIndices,
                                            const Board *board);

[[nodiscard]] std::vector<EncodedMoveScore>
filterPolicyThenGetMovesAndProbabilities(const torch::Tensor &policy, const Board *board);

[[nodiscard]] std::vector<MoveScore> filterPolicyThenGetMoveScores(const float *policyData,
                                                                   const Board *board);

using ChessAction = Move;

[[nodiscard]] inline int chess_action_id(const ChessAction action, const Board &position) {
    return encodeMove(action, &position);
}

[[nodiscard]] inline std::vector<ChessAction>
decode_chess_actions(const std::vector<int> &action_ids, const Board &position) {
    return decodeMoves(action_ids, &position);
}
