#pragma once

#include "MCTS.hpp"

#include "../DirectInference.hpp"

#include <deque>

class DirectSelfPlaySearch {
public:
    DirectSelfPlaySearch(const InferenceClientParams &clientParameters,
                         const MCTSParams &searchParameters,
                         const DirectSelfPlayInferenceParams &inferenceParameters);

    [[nodiscard]] MCTSResults search(const std::vector<MCTSBoard> &boards, bool collectStatistics);
    [[nodiscard]] InferenceResult evaluate(const Board &board);
    [[nodiscard]] InferenceStatistics inferenceStatistics() const;

private:
    struct RootTask {
        MCTSRoot root;
        uint32 visit_limit;
        uint32 in_flight = 0;
        bool noise_pending;
    };

    struct PendingLeaf {
        std::size_t task_index;
        NodeIndex node_index;
        bool counts_as_search;
    };

    struct PendingBatch {
        std::size_t slot_index;
        std::vector<PendingLeaf> leaves;
    };

    MCTSParams m_searchParameters;
    DirectSelfPlayInferenceParams m_inferenceParameters;
    std::vector<std::unique_ptr<DirectInferencePipeline>> m_workers;
    std::vector<std::deque<PendingBatch>> m_pending;
    std::size_t m_nextWorker = 0;
    std::size_t m_nextTask = 0;
    std::uint64_t m_evaluations = 0;
    std::uint64_t m_modelCalls = 0;
    std::uint64_t m_modelPositions = 0;
    std::vector<std::size_t> m_batchHistogram;
    std::uint64_t m_searchWallNanoseconds = 0;
    std::uint64_t m_selectionNanoseconds = 0;
    std::uint64_t m_encodingNanoseconds = 0;
    std::uint64_t m_resultProcessingNanoseconds = 0;
    std::uint64_t m_backupNanoseconds = 0;
    std::uint64_t m_waitNanoseconds = 0;

    [[nodiscard]] std::optional<std::size_t> freeWorker() const;
    [[nodiscard]] std::optional<std::size_t> readyWorker(std::size_t firstWorker) const;
    [[nodiscard]] std::optional<std::size_t> schedulableTask(const std::vector<RootTask> &tasks);
    [[nodiscard]] std::optional<NodeIndex> selectLeaf(MCTSRoot &root) const;
    void addNoise(MCTSRoot &root) const;
    void completeWorker(std::vector<RootTask> &tasks, std::size_t workerIndex,
                        uint64 &completedSearches);
    void cancelPending(std::vector<RootTask> &tasks) noexcept;
    void recordBatch(std::size_t batchSize);
};
