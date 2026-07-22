#include "InteractiveSearch.hpp"

#include "InferenceResultProcessing.hpp"
#include "MoveEncoding.hpp"

namespace {
constexpr std::size_t ENCODED_BOARD_SIZE = BOARD_C * BOARD_LEN * BOARD_LEN;
}

InteractiveSearchParams::InteractiveSearchParams(const float explorationConstant,
                                                 const int inferenceWorkers,
                                                 const int inferenceBatchSize)
    : exploration_constant(explorationConstant), inference_workers(inferenceWorkers),
      inference_batch_size(inferenceBatchSize) {
    if (!std::isfinite(exploration_constant) || exploration_constant < 0.0F) {
        throw std::invalid_argument("exploration_constant must be finite and non-negative");
    }
    if (inference_workers <= 0) {
        throw std::invalid_argument("inference_workers must be positive");
    }
    if (inference_batch_size <= 0) {
        throw std::invalid_argument("inference_batch_size must be positive");
    }
}

InteractiveSearch::InteractiveSearch(const InferenceClientParams &clientParameters,
                                     const InteractiveSearchParams &searchParameters)
    : m_parameters(searchParameters),
      m_pending(static_cast<std::size_t>(searchParameters.inference_workers)),
      m_batchHistogram(static_cast<std::size_t>(searchParameters.inference_batch_size) + 1, 0) {
    m_workers.reserve(static_cast<std::size_t>(searchParameters.inference_workers));
    for (int worker = 0; worker < searchParameters.inference_workers; ++worker) {
        m_workers.push_back(std::make_unique<DirectInferencePipeline>(
            clientParameters.currentModelPath, clientParameters.device, clientParameters.device_id,
            static_cast<std::size_t>(searchParameters.inference_batch_size), 2, true));
    }
}

InferenceResult InteractiveSearch::decode(const torch::Tensor &policy, const torch::Tensor &outcome,
                                          const Board &board) {
    if (policy.device().is_cuda() || policy.scalar_type() != torch::kFloat32 || policy.dim() != 1 ||
        policy.numel() != ACTION_SIZE || !policy.is_contiguous()) {
        throw std::runtime_error("Inference model policy output must contain one float per action");
    }
    if (outcome.device().is_cuda() || outcome.scalar_type() != torch::kFloat32 ||
        outcome.dim() != 1 || outcome.numel() != 3 || !outcome.is_contiguous()) {
        throw std::runtime_error("Inference model WDL output must be three probabilities");
    }
    return processInferenceResult(policy.data_ptr<float>(), outcome.data_ptr<float>(), board);
}

InferenceResult InteractiveSearch::evaluate(const Board &board) {
    DirectInferencePipeline &worker = *m_workers.front();
    const DirectInferencePipeline::WritableBatch writable = worker.acquireWritableBatch();
    encodeBoardInto(board, writable.data);
    worker.submit(writable.slotIndex, 1);
    bool outputReady = false;
    try {
        DirectInferenceOutput output = worker.waitCompleted(writable.slotIndex);
        outputReady = true;
        InferenceResult result = decode(output.policies[0], output.outcomes[0], board);
        worker.release(writable.slotIndex);
        recordBatch(1, std::chrono::microseconds(0));
        return result;
    } catch (...) {
        if (outputReady) {
            worker.release(writable.slotIndex);
        }
        throw;
    }
}

bool InteractiveSearch::mayIssue(
    const std::optional<std::chrono::steady_clock::time_point> &deadline,
    const std::optional<int> &searchLimit, const int claimed) const {
    if (searchLimit.has_value() && claimed >= *searchLimit) {
        return false;
    }
    if (!deadline.has_value()) {
        return true;
    }
    const auto safetyMargin =
        std::max(std::chrono::microseconds(5'000), m_inferenceLatencyEstimate * 2);
    return std::chrono::steady_clock::now() + safetyMargin < *deadline;
}

std::optional<std::size_t> InteractiveSearch::freeWorker() const {
    for (std::size_t offset = 0; offset < m_workers.size(); ++offset) {
        const std::size_t index = (m_nextWorker + offset) % m_workers.size();
        if (!m_pending[index].has_value()) {
            return index;
        }
    }
    return std::nullopt;
}

void InteractiveSearch::recordBatch(const std::size_t batchSize,
                                    const std::chrono::steady_clock::duration inferenceDuration) {
    ++m_modelCalls;
    m_modelPositions += batchSize;
    m_evaluations += batchSize;
    if (batchSize < m_batchHistogram.size()) {
        ++m_batchHistogram[batchSize];
    }
    if (inferenceDuration > std::chrono::steady_clock::duration::zero()) {
        const auto measured =
            std::chrono::duration_cast<std::chrono::microseconds>(inferenceDuration);
        m_inferenceLatencyEstimate = (m_inferenceLatencyEstimate * 7 + measured) / 8;
    }
}

void InteractiveSearch::completeWorker(EvalSearchTree &tree, const std::size_t workerIndex,
                                       int &completed) {
    std::optional<PendingBatch> &pending = m_pending[workerIndex];
    if (!pending.has_value()) {
        throw std::logic_error("Cannot complete an idle direct inference worker");
    }
    DirectInferencePipeline &worker = *m_workers[workerIndex];
    std::size_t processed = 0;
    bool outputReady = false;
    try {
        const auto waitStartedAt = std::chrono::steady_clock::now();
        DirectInferenceOutput output = worker.waitCompleted(pending->slot_index);
        m_waitNanoseconds +=
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now() - waitStartedAt)
                                           .count());
        outputReady = true;
        const float *policyData = output.policies.data_ptr<float>();
        const float *outcomeData = output.outcomes.data_ptr<float>();
        for (; processed < pending->leaves.size(); ++processed) {
            const EvalNodeIndex leafIndex = pending->leaves[processed];
            EvalSearchNode &leaf = tree.node(leafIndex);
            const auto processingStartedAt = std::chrono::steady_clock::now();
            const InferenceResult inferenceResult = processInferenceResult(
                policyData + processed * ACTION_SIZE, outcomeData + processed * 3, leaf.board);
            m_resultProcessingNanoseconds += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - processingStartedAt)
                    .count());
            const auto backupStartedAt = std::chrono::steady_clock::now();
            tree.expand(leafIndex, inferenceResult.moves, inferenceResult.outcome);
            tree.completeReservation(leafIndex, inferenceResult.value());
            m_backupNanoseconds +=
                static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                               std::chrono::steady_clock::now() - backupStartedAt)
                                               .count());
            ++completed;
        }
        worker.release(pending->slot_index);
        recordBatch(pending->leaves.size(),
                    std::chrono::steady_clock::now() - pending->submitted_at);
        pending.reset();
    } catch (...) {
        if (outputReady) {
            worker.release(pending->slot_index);
        }
        for (std::size_t index = processed; index < pending->leaves.size(); ++index) {
            tree.cancelReservation(pending->leaves[index]);
        }
        pending.reset();
        throw;
    }
}

void InteractiveSearch::cancelPending(EvalSearchTree &tree) noexcept {
    for (std::size_t workerIndex = 0; workerIndex < m_pending.size(); ++workerIndex) {
        std::optional<PendingBatch> &pending = m_pending[workerIndex];
        if (!pending.has_value()) {
            continue;
        }
        try {
            static_cast<void>(m_workers[workerIndex]->waitCompleted(pending->slot_index));
            m_workers[workerIndex]->release(pending->slot_index);
        } catch (...) {
        }
        for (const EvalNodeIndex leafIndex : pending->leaves) {
            try {
                tree.cancelReservation(leafIndex);
            } catch (...) {
            }
        }
        pending.reset();
    }
}

InteractiveSearchResult
InteractiveSearch::search(EvalSearchTree &tree,
                          const std::optional<std::chrono::steady_clock::time_point> deadline,
                          const std::optional<int> searchLimit) {
    if (searchLimit.has_value() && *searchLimit <= 0) {
        throw std::invalid_argument("search_limit must be positive");
    }
    int claimed = 0;
    int completed = 0;
    std::size_t completionCursor = 0;
    const auto searchStartedAt = std::chrono::steady_clock::now();

    try {
        while (true) {
            const bool issuing = mayIssue(deadline, searchLimit, claimed);
            const std::optional<std::size_t> availableWorker =
                issuing ? freeWorker() : std::nullopt;
            if (availableWorker.has_value()) {
                const std::size_t workerIndex = *availableWorker;
                DirectInferencePipeline &worker = *m_workers[workerIndex];
                const DirectInferencePipeline::WritableBatch writable =
                    worker.acquireWritableBatch();
                std::vector<EvalNodeIndex> leaves;
                leaves.reserve(static_cast<std::size_t>(m_parameters.inference_batch_size));
                try {
                    while (leaves.size() <
                               static_cast<std::size_t>(m_parameters.inference_batch_size) &&
                           mayIssue(deadline, searchLimit, claimed)) {
                        const auto selectionStartedAt = std::chrono::steady_clock::now();
                        const EvalNodeIndex leaf =
                            tree.selectLeaf(m_parameters.exploration_constant);
                        m_selectionNanoseconds += static_cast<std::uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - selectionStartedAt)
                                .count());
                        if (leaf == INVALID_EVAL_NODE_INDEX) {
                            break;
                        }
                        if (tree.node(leaf).isTerminal()) {
                            tree.backPropagate(leaf, getBoardResultScore(tree.node(leaf).board));
                            ++claimed;
                            ++completed;
                            continue;
                        }
                        tree.reserveLeaf(leaf);
                        const auto encodingStartedAt = std::chrono::steady_clock::now();
                        encodeBoardInto(tree.node(leaf).board,
                                        writable.data + leaves.size() * ENCODED_BOARD_SIZE);
                        m_encodingNanoseconds += static_cast<std::uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - encodingStartedAt)
                                .count());
                        leaves.push_back(leaf);
                        ++claimed;
                    }
                } catch (...) {
                    for (const EvalNodeIndex leaf : leaves) {
                        tree.cancelReservation(leaf);
                    }
                    worker.discardWritableBatch(writable.slotIndex);
                    throw;
                }

                if (leaves.empty()) {
                    worker.discardWritableBatch(writable.slotIndex);
                } else {
                    worker.submit(writable.slotIndex, leaves.size());
                    m_pending[workerIndex] = PendingBatch{writable.slotIndex, std::move(leaves),
                                                          std::chrono::steady_clock::now()};
                    m_nextWorker = (workerIndex + 1) % m_workers.size();
                    continue;
                }
            }

            const bool hasPending =
                std::ranges::any_of(m_pending, [](const auto &batch) { return batch.has_value(); });
            if (!hasPending) {
                if (!mayIssue(deadline, searchLimit, claimed)) {
                    break;
                }
                throw std::logic_error("No selectable leaf and no pending inference");
            }
            while (!m_pending[completionCursor].has_value()) {
                completionCursor = (completionCursor + 1) % m_pending.size();
            }
            completeWorker(tree, completionCursor, completed);
            completionCursor = (completionCursor + 1) % m_pending.size();
        }
    } catch (...) {
        cancelPending(tree);
        assert(tree.evaluatingNodeCount() == 0);
        assert(tree.totalVirtualLoss() == 0);
        throw;
    }

    assert(tree.evaluatingNodeCount() == 0);
    assert(tree.totalVirtualLoss() == 0);
    m_searchWallNanoseconds +=
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - searchStartedAt)
                                       .count());
    const EvalSearchStatistics &statistics = tree.rootStatistics();
    const float result = statistics.visits == 0
                             ? 0.0F
                             : statistics.result_sum / static_cast<float>(statistics.visits);
    return {result, completed};
}

InferenceStatistics InteractiveSearch::inferenceStatistics() const {
    InferenceStatistics statistics;
    statistics.evaluations = m_evaluations;
    statistics.modelInferenceCalls = m_modelCalls;
    statistics.modelInferencePositions = m_modelPositions;
    statistics.modelBatchSizeHistogram = m_batchHistogram;
    statistics.averageNumberOfPositionsInInferenceCall =
        m_modelCalls == 0 ? 0.0F
                          : static_cast<float>(m_modelPositions) / static_cast<float>(m_modelCalls);
    statistics.treeSelectionNanoseconds = m_selectionNanoseconds;
    statistics.boardEncodingNanoseconds = m_encodingNanoseconds;
    statistics.resultProcessingNanoseconds = m_resultProcessingNanoseconds;
    statistics.treeBackupNanoseconds = m_backupNanoseconds;
    statistics.treeOwnerWaitNanoseconds = m_waitNanoseconds;
    for (const std::unique_ptr<DirectInferencePipeline> &worker : m_workers) {
        statistics.directInferenceNanoseconds += worker->inferenceNanoseconds();
    }
    const std::uint64_t availableWorkerNanoseconds =
        m_searchWallNanoseconds * static_cast<std::uint64_t>(m_workers.size());
    statistics.directWorkerUtilization =
        availableWorkerNanoseconds == 0
            ? 0.0F
            : static_cast<float>(statistics.directInferenceNanoseconds) /
                  static_cast<float>(availableWorkerNanoseconds);
    return statistics;
}
