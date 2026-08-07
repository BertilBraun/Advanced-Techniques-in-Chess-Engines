#pragma once

#include "common.hpp"

#include "SearchTree.hpp"

#include "InferenceClientTypes.hpp"
#include <shared_mutex>

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
    float averagePolicySearchKLDivergence = 0.0f;
    float topMoveDisagreement = 0.0f;
    float selectedMovePriorRank = 0.0f;
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
         DirectSelfPlayInferenceParams directInferenceParams,
         uint64 initialModelVersion = 0);
    ~MCTS();

    [[nodiscard]] MCTSResults search(const std::vector<MCTSBoard> &boards,
                                     bool collectStatistics = false);
    [[nodiscard]] MCTSRoot newRoot(const std::string &fen) const;
    [[nodiscard]] MCTSRoot newRoot(Board board) const;
    [[nodiscard]] uint32 arenaCapacity() const;
    [[nodiscard]] uint64 modelVersion() const;

    [[nodiscard]] std::pair<InferenceStatistics, TimeInfo> getInferenceStatistics();
    void refreshModel(uint64 modelVersion, const std::string &modelPath);
    [[nodiscard]] bool updateSearchSchedule(const MCTSParams &mctsArgs);
    [[nodiscard]] std::vector<InferenceResult>
    inferenceBatch(const std::vector<const Board *> &boards);
    [[nodiscard]] std::vector<std::uintptr_t> directWorkerIdentityTokens() const;

private:
    InferenceClientParams m_clientArgs;
    MCTSParams m_args;
    uint32 m_arenaCapacity;
    uint64 m_modelVersion;
    DirectSelfPlayInferenceParams m_directInferenceParams;
    std::unique_ptr<DirectSelfPlaySearch> m_directSearch;
    mutable std::shared_mutex m_operationMutex;

    [[nodiscard]] std::vector<InferenceResult>
    inferenceBatchUnlocked(const std::vector<const Board *> &boards);
};
