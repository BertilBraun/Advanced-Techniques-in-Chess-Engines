#pragma once

#include "common.hpp"

#include "../BoardEncoding.hpp"
#include "../MoveEncoding.hpp"
#include "MCTSNode.hpp"
#include "VisitCounts.hpp"

#include "../InferenceClient.hpp"

struct MCTSResult {
    float result;
    VisitCounts visits;

    MCTSResult(float result_, const VisitCounts &visits_) : result(result_), visits(visits_) {}
};

class MCTS {
public:
    MCTS(InferenceClient *client, const MCTSParams &args, TensorBoardLogger *logger)
        : client(client), args(args), m_logger(logger) {}

    std::vector<MCTSResult> search(std::vector<Board> &boards) const;

    // This method performs several iterations of tree search in parallel.
    void parallel_iterate(std::vector<MCTSNode> &roots) const;

private:
    InferenceClient *client;
    MCTSParams args;
    TensorBoardLogger *m_logger;

    // Get policy moves with added Dirichlet noise.
    std::vector<std::vector<MoveScore>> _get_policy_with_noise(std::vector<Board> &boards) const;

    // Add Dirichlet noise to a vector of MoveScore.
    std::vector<MoveScore> _add_noise(const std::vector<MoveScore> &moves) const;

    // Traverse the tree to find the best child or, if the node is terminal,
    // back-propagate the board’s result.
    std::optional<MCTSNode *> _get_best_child_or_back_propagate(MCTSNode &root,
                                                                float c_param) const;

    void _logMCTSStatistics(const std::vector<MCTSNode> &roots) const;
};
