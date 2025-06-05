#pragma once

#include "common.hpp"

#include "MCTSNode.hpp"

#include "../InferenceClient.hpp"

typedef std::pair<Move, int> VisitCount;
typedef std::vector<VisitCount> VisitCounts;

struct MCTSResult {
    float result;
    VisitCounts visits;
    std::vector<MCTSNode> children;
    bool fullSearch;
};

struct MCTSStatistics {
    float averageDepth = 0.0f;        // Average depth of the search trees.
    float averageEntropy = 0.0f;      // Average entropy of the visit counts.
    float averageKLDivergence = 0.0f; // Average KL divergence of the visit counts.
};

struct MCTSResults {
    std::vector<MCTSResult> results;
    MCTSStatistics mctsStats;
};

class MCTS {
public:
    MCTS(const InferenceClientParams &clientArgs, const MCTSParams &mctsArgs)
        : m_client(clientArgs), m_args(mctsArgs) {}

    MCTSResults search(std::vector<std::tuple<std::string, MCTSNode, bool>> &boards);

    InferenceStatistics getInferenceStatistics() const;

private:
    InferenceClient m_client;
    MCTSParams m_args;

    // This method performs several iterations of tree search in parallel.
    void parallelIterate(std::vector<MCTSNode> &roots);

    // Get policy moves with added Dirichlet noise.
    std::vector<std::vector<MoveScore>>
    getPolicyWithNoise(const std::vector<const Board &> &boards);

    // Add Dirichlet noise to a vector of MoveScore.
    std::vector<MoveScore> addNoise(const std::vector<MoveScore> &moves) const;
};
