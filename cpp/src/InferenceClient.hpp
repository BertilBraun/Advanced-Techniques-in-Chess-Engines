#pragma once

#include "common.hpp"

#include "BoardEncoding.hpp"
#include "MoveEncoding.hpp"

typedef std::vector<std::pair<std::vector<MoveScore>, float>> InferenceResult;

class InferenceClient {
public:
    // The inference_batch function takes a vector of boards and returns for each board a pair:
    //   - A vector of MoveScore (the move probabilities)
    //   - A float value (the board evaluation)
    InferenceResult inference_batch(std::vector<Board> &boards) const {
        torch::NoGradGuard no_grad; // Disable gradient calculation equivalent to torch.no_grad()

        // TODO model.eval()

        // TODO periodically (every 5sec or so, check for a new model to load)

        InferenceResult results;
        // TODO: Dummy implementation: for each board, return 5 moves with random policies and a
        // dummy value.
        for (auto &board : boards) {
            auto legalMoves = board.legalMoves();
            std::vector<MoveScore> moves;
            for (auto move : legalMoves) {
                moves.emplace_back(encodeMove(move),
                                   1.0f / legalMoves.size()); // Dummy uniform policy
            }
            results.emplace_back(moves, 0.0f);
        }
        return results;
    }
};
