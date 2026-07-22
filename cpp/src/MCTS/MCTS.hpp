#pragma once

#include "common.hpp"

#include "SearchTree.hpp"

#include "../InferenceClient.hpp"
#include "../NonCachingInferenceClient.hpp"
#include "util/ThreadPool.h"
#include <variant>

class DirectSelfPlaySearch;

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

struct DirectSelfPlayInferenceParams {
    int inference_workers;
    int inference_batch_size;
    int outstanding_batches_per_worker;

    DirectSelfPlayInferenceParams(int inferenceWorkers, int inferenceBatchSize,
                                  int outstandingBatchesPerWorker = 2);
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
    uint64 searchesCompleted;
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
         bool useInferenceCache = true,
         std::optional<DirectSelfPlayInferenceParams> directInferenceParams = std::nullopt);
    ~MCTS();

    [[nodiscard]] MCTSResults search(const std::vector<MCTSBoard> &boards,
                                     bool collectStatistics = false);
    [[nodiscard]] MCTSRoot newRoot(const std::string &fen) const;
    [[nodiscard]] MCTSRoot newRoot(Board board) const;
    [[nodiscard]] uint32 arenaCapacity() const { return m_arenaCapacity; }

    [[nodiscard]] std::pair<InferenceStatistics, TimeInfo> getInferenceStatistics();
    void update(const std::string &modelPath, const MCTSParams &mctsArgs);
    [[nodiscard]] std::vector<InferenceResult>
    inferenceBatch(const std::vector<const Board *> &boards);

private:
    using InferenceClientVariant = std::variant<std::monostate, std::unique_ptr<InferenceClient>,
                                                std::unique_ptr<NonCachingInferenceClient>>;

    InferenceClientVariant m_client;
    InferenceClientParams m_clientArgs;
    MCTSParams m_args;
    ThreadPool m_threadPool;
    uint32 m_arenaCapacity;
    std::optional<DirectSelfPlayInferenceParams> m_directInferenceParams;
    std::unique_ptr<DirectSelfPlaySearch> m_directSearch;

    [[nodiscard]] MCTSResults searchGames(const std::vector<MCTSBoard> &boards);
    void parallelIterate(const std::vector<MCTSRoot> &roots);
    void addNoise(MCTSRoot &root) const;
    [[nodiscard]] std::optional<NodeIndex> getBestChildOrBackPropagate(MCTSRoot &root,
                                                                       float cParam) const;
};
