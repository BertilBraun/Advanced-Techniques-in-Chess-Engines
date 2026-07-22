#pragma once

#include "DirectInference.hpp"
#include "MCTS/EvalSearchTree.hpp"

struct InteractiveSearchParams {
    float exploration_constant;
    int inference_workers;
    int inference_batch_size;

    InteractiveSearchParams(float explorationConstant, int inferenceWorkers,
                            int inferenceBatchSize);
};

struct InteractiveSearchResult {
    float result;
    int completed_searches;
};

class InteractiveSearch {
public:
    InteractiveSearch(const InferenceClientParams &clientParameters,
                      const InteractiveSearchParams &searchParameters);

    InteractiveSearch(const InteractiveSearch &) = delete;
    InteractiveSearch &operator=(const InteractiveSearch &) = delete;

    [[nodiscard]] InferenceResult evaluate(const Board &board);
    [[nodiscard]] InteractiveSearchResult
    search(EvalSearchTree &tree, std::optional<std::chrono::steady_clock::time_point> deadline,
           std::optional<int> searchLimit);
    [[nodiscard]] InferenceStatistics inferenceStatistics() const;

private:
    struct PendingBatch {
        std::size_t slot_index;
        std::vector<EvalNodeIndex> leaves;
        std::chrono::steady_clock::time_point submitted_at;
    };

    InteractiveSearchParams m_parameters;
    std::vector<std::unique_ptr<DirectInferencePipeline>> m_workers;
    std::vector<std::optional<PendingBatch>> m_pending;
    std::size_t m_nextWorker = 0;
    std::uint64_t m_evaluations = 0;
    std::uint64_t m_modelCalls = 0;
    std::uint64_t m_modelPositions = 0;
    std::vector<std::size_t> m_batchHistogram;
    std::chrono::microseconds m_inferenceLatencyEstimate{20'000};
    std::uint64_t m_searchWallNanoseconds = 0;
    std::uint64_t m_selectionNanoseconds = 0;
    std::uint64_t m_encodingNanoseconds = 0;
    std::uint64_t m_resultProcessingNanoseconds = 0;
    std::uint64_t m_backupNanoseconds = 0;
    std::uint64_t m_waitNanoseconds = 0;

    [[nodiscard]] static InferenceResult decode(const torch::Tensor &policy,
                                                const torch::Tensor &outcome, const Board &board);
    [[nodiscard]] bool
    mayIssue(const std::optional<std::chrono::steady_clock::time_point> &deadline,
             const std::optional<int> &searchLimit, int claimed) const;
    [[nodiscard]] std::optional<std::size_t> freeWorker() const;
    void completeWorker(EvalSearchTree &tree, std::size_t workerIndex, int &completed);
    void cancelPending(EvalSearchTree &tree) noexcept;
    void recordBatch(std::size_t batchSize, std::chrono::steady_clock::duration inferenceDuration);
};
