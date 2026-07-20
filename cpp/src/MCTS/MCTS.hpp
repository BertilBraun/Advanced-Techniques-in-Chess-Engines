#pragma once

#include "common.hpp"

#include "SearchTree.hpp"

#include "../InferenceClient.hpp"
#include "../NonCachingInferenceClient.hpp"
#include "util/ThreadPool.h"
#include <variant>

using VisitCount = std::pair<int, int>;
using VisitCounts = std::vector<VisitCount>;

struct MCTSParams {
    int num_parallel_searches;
    uint32 num_full_searches;
    uint32 num_fast_searches;
    float c_param;
    float dirichlet_alpha;
    float dirichlet_epsilon;
    uint8 min_visit_count;
    uint8 num_threads;

    MCTSParams(int num_parallel_searches, uint32 num_full_searches, uint32 num_fast_searches,
               float c_param, float dirichlet_alpha, float dirichlet_epsilon, uint8 min_visit_count,
               uint8 num_threads);

    [[nodiscard]] uint32 arenaCapacity() const;
};

struct MCTSResult {
    float result;
    VisitCounts visits;
    MCTSRoot root;
};

struct MCTSStatistics {
    float averageDepth = 0.0f;
    float averageEntropy = 0.0f;
    float averageKLDivergence = 0.0f;
};

struct MCTSResults {
    std::vector<MCTSResult> results;
    MCTSStatistics mctsStats;
};

struct MCTSBoard {
    MCTSRoot root;
    bool should_run_full_search;

    MCTSBoard(MCTSRoot root, bool shouldRunFullSearch)
        : root(std::move(root)), should_run_full_search(shouldRunFullSearch) {}
};

class MCTS {
public:
    MCTS(const InferenceClientParams &clientArgs, const MCTSParams &mctsArgs,
         bool useInferenceCache = true);

    [[nodiscard]] MCTSResults search(const std::vector<MCTSBoard> &boards,
                                     bool collectStatistics = false);
    [[nodiscard]] MCTSRoot newRoot(const std::string &fen) const;
    [[nodiscard]] MCTSRoot newRoot(Board board) const;
    [[nodiscard]] uint32 arenaCapacity() const { return m_arenaCapacity; }

    [[nodiscard]] std::pair<InferenceStatistics, TimeInfo> getInferenceStatistics();
    [[nodiscard]] std::vector<InferenceResult>
    inferenceBatch(const std::vector<const Board *> &boards);

private:
    using InferenceClientVariant =
        std::variant<std::unique_ptr<InferenceClient>, std::unique_ptr<NonCachingInferenceClient>>;

    InferenceClientVariant m_client;
    MCTSParams m_args;
    ThreadPool m_threadPool;
    uint32 m_arenaCapacity;

    [[nodiscard]] std::vector<MCTSResult> searchGames(const std::vector<MCTSBoard> &boards);
    void parallelIterate(const std::vector<MCTSRoot> &roots);
    void addNoise(MCTSRoot &root) const;
    [[nodiscard]] std::optional<NodeIndex> getBestChildOrBackPropagate(MCTSRoot &root,
                                                                       float cParam) const;
};
