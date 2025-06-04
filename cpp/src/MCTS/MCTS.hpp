#pragma once

#include "common.hpp"

#include "MCTSNode.hpp"
#include "VisitCounts.hpp"

#include "../InferenceClient.hpp"

struct MCTSResult {
    float result;
    VisitCounts visits;
    std::vector<MCTSNode> children;
    bool fullSearch;

    MCTSResult(float result_, const VisitCounts &visits_, std::vector<MCTSNode> children_ = {},
               bool fullSearch_ = false)
        : result(result_), visits(visits_), children(std::move(children_)), fullSearch(fullSearch_) {}
};

struct MCTSStatistics {
    float averageDepth;
    float averageEntropy;
    float averageKLDivergence;
};

struct InferenceStatistics {
    float cacheHitRate;
    size_t uniquePositions;
    size_t cacheSizeMB;
    std::vector<float> nnOutputValueDistribution;
};

struct MCTSResults {
    std::vector<MCTSResult> results;
    MCTSStatistics mctsStats;
};

class MCTS {
public:
    MCTS(const InferenceClientParams& clientArgs, const MCTSParams &mctsArgs)
        : m_client(clientArgs), m_args(mctsArgs) {}

    std::vector<MCTSResult> search(std::vector<std::pair<std::string, std::shared_ptr<MCTSNode>>> &boards) const;

    InferenceStatistics getInferenceStatistics() const;
private:
    InferenceClient m_client;
    MCTSParams m_args;

    // This method performs several iterations of tree search in parallel.
    void parallelIterate(std::vector<MCTSNode> &roots) const;

    // Get policy moves with added Dirichlet noise.
    std::vector<std::vector<MoveScore>> getPolicyWithNoise(std::vector<Board> &boards) const;

    // Add Dirichlet noise to a vector of MoveScore.
    std::vector<MoveScore> addNoise(const std::vector<MoveScore> &moves) const;

    // Traverse the tree to find the best child or, if the node is terminal,
    // back-propagate the board’s result.
    std::optional<MCTSNode *> getBestChildOrBackPropagate(MCTSNode &root, float c_param) const;

    void logMctsStatistics(const std::vector<MCTSNode> &roots) const;
};
