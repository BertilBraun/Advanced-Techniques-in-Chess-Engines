#pragma once

#include "search/InferencePipeline.hpp"
#include "search/SearchInference.hpp"
#include "search/SearchTree.hpp"
#include "search/SearchTypes.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

template <typename Game> class BatchedGameSearch {
public:
    using Root = GameSearchRoot<Game>;
    using Tree = GameSearchTree<Game>;

    BatchedGameSearch(const std::string &modelPath, const InferenceDevice device,
                      const int deviceId, const BatchedInferenceParameters inferenceParameters,
                      const BatchedSearchParameters searchParameters,
                      const std::uint64_t modelGeneration, const bool resetTreesOnRefresh = true,
                      const float turnDiscount = 1.0F)
        : m_inferenceParameters(inferenceParameters), m_searchParameters(searchParameters),
          m_dimensions(Game::inferenceDimensions()), m_modelGeneration(modelGeneration),
          m_pending(inferenceParameters.workers), m_randomEngine(std::random_device{}()),
          m_batchHistogram(inferenceParameters.batch_size + 1, 0),
          m_resetTreesOnRefresh(resetTreesOnRefresh), m_turnDiscount(turnDiscount) {
        m_workers.reserve(inferenceParameters.workers);
        for (std::size_t worker = 0; worker < inferenceParameters.workers; ++worker) {
            m_workers.push_back(std::make_unique<InferencePipeline>(
                modelPath, device, deviceId, inferenceParameters.batch_size,
                std::max<std::size_t>(2, inferenceParameters.outstanding_batches_per_worker),
                true, m_dimensions));
        }
    }

    [[nodiscard]] Root newRoot(typename Game::Position position,
                               const std::size_t maximumCapacity = 0) {
        Root root(std::move(position), m_searchParameters.tree_capacity, maximumCapacity,
                  m_turnDiscount);
        m_trees.push_back(root.sharedTree());
        return root;
    }

    [[nodiscard]] std::vector<GameSearchResult> search(std::vector<Root> &roots,
                                                       const std::uint32_t simulations) {
        std::vector<GameSearchRequest<Game>> requests;
        requests.reserve(roots.size());
        for (Root &root : roots) {
            requests.push_back({root, root.visits() + simulations, true});
        }
        return searchDetailed(requests).results;
    }

    [[nodiscard]] GameSearchBatchResult
    searchDetailed(const std::vector<GameSearchRequest<Game>> &requests) {
        const auto searchStartedAt = std::chrono::steady_clock::now();
        const bool hasInvalidRequest = std::ranges::any_of(
            requests, [](const GameSearchRequest<Game> &request) {
                return request.visit_limit == 0;
            });
        if (requests.empty() || hasInvalidRequest) {
            throw std::invalid_argument("Batched search requires roots and simulations");
        }
        std::vector<RootTask> tasks;
        tasks.reserve(requests.size());
        for (const GameSearchRequest<Game> &request : requests) {
            Root root = request.root;
            const std::uint32_t startingVisits = root.visits();
            const std::uint32_t visitLimit = request.visit_limit;
            root.tree().prepareForSearch(visitLimit, m_searchParameters.parallel_searches);
            tasks.push_back({root, startingVisits, visitLimit, 0,
                             request.add_root_noise && !root.tree().root().expanded(),
                             request.count_root_initialization});
            if (request.add_root_noise && root.tree().root().expanded()) {
                addNoise(tasks.back().root);
            }
        }
        std::size_t completionCursor = 0;
        try {
            while (true) {
                const std::optional<std::size_t> workerIndex = freeWorker();
                if (workerIndex.has_value() && issueBatch(tasks, *workerIndex)) {
                    continue;
                }
                if (!hasPending()) {
                    break;
                }
                const std::optional<std::size_t> completed = readyWorker(completionCursor);
                if (completed.has_value()) {
                    completionCursor = *completed;
                } else {
                    while (m_pending[completionCursor].empty()) {
                        completionCursor = (completionCursor + 1) % m_pending.size();
                    }
                }
                completeWorker(tasks, completionCursor);
                completionCursor = (completionCursor + 1) % m_pending.size();
            }
        } catch (...) {
            cancelPending(tasks);
            throw;
        }
        std::vector<GameSearchResult> results;
        results.reserve(tasks.size());
        for (const RootTask &task : tasks) {
            const Root &root = task.root;
            const auto &node = root.tree().root();
            GameSearchResult result{node.visits == 0 ? 0.0F : node.value_sum / node.visits, {}};
            result.visits.reserve(node.children.size());
            for (const auto &edge : node.children) {
                result.visits.push_back({Game::actionId(edge.action, node.position), edge.visits});
            }
            results.push_back(std::move(result));
        }
        const std::uint64_t completed = std::accumulate(
            tasks.begin(), tasks.end(), std::uint64_t{0},
            [](const std::uint64_t count, const RootTask &task) {
                return count + task.root.visits() - task.starting_visits;
            });
        m_searchWallNanoseconds +=
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now() - searchStartedAt)
                                           .count());
        return {std::move(results), completed};
    }

    [[nodiscard]] std::vector<SearchInferenceResult<Game>>
    evaluate(const std::vector<typename Game::Position> &positions) {
        std::vector<SearchInferenceResult<Game>> results;
        results.reserve(positions.size());
        std::size_t offset = 0;
        InferencePipeline &worker = *m_workers.front();
        constexpr std::size_t encodedSize =
            static_cast<std::size_t>(Game::inferenceDimensions().channels) *
            static_cast<std::size_t>(Game::inferenceDimensions().rows) *
            static_cast<std::size_t>(Game::inferenceDimensions().columns);
        while (offset < positions.size()) {
            const std::size_t batchSize =
                std::min(m_inferenceParameters.batch_size, positions.size() - offset);
            const InferencePipeline::WritableBatch writable = worker.acquireWritableBatch();
            for (std::size_t row = 0; row < batchSize; ++row) {
                Game::encodeInputInto(positions[offset + row],
                                      writable.data + row * encodedSize);
            }
            worker.submit(writable.slotIndex, batchSize);
            bool outputReady = false;
            try {
                const InferenceOutput output = worker.waitCompleted(writable.slotIndex);
                outputReady = true;
                const float *policies = output.policies.data_ptr<float>();
                const float *outcomes = output.outcomes.data_ptr<float>();
                for (std::size_t row = 0; row < batchSize; ++row) {
                    results.push_back(processSearchInference<Game>(
                        policies + row * static_cast<std::size_t>(m_dimensions.actions),
                        outcomes + row * static_cast<std::size_t>(m_dimensions.outcomes),
                        positions[offset + row]));
                }
                worker.release(writable.slotIndex);
                recordBatch(batchSize);
            } catch (...) {
                if (outputReady) {
                    worker.release(writable.slotIndex);
                }
                throw;
            }
            offset += batchSize;
        }
        return results;
    }

    [[nodiscard]] InferenceStatistics inferenceStatistics() const {
        InferenceStatistics statistics;
        statistics.evaluations = m_evaluations;
        statistics.modelInferenceCalls = m_modelCalls;
        statistics.modelInferencePositions = m_modelPositions;
        statistics.modelBatchSizeHistogram = m_batchHistogram;
        statistics.averageNumberOfPositionsInInferenceCall =
            m_modelCalls == 0
                ? 0.0F
                : static_cast<float>(m_modelPositions) / static_cast<float>(m_modelCalls);
        statistics.treeSelectionNanoseconds = m_selectionNanoseconds;
        statistics.boardEncodingNanoseconds = m_encodingNanoseconds;
        statistics.resultProcessingNanoseconds = m_resultProcessingNanoseconds;
        statistics.treeBackupNanoseconds = m_backupNanoseconds;
        statistics.treeOwnerWaitNanoseconds = m_waitNanoseconds;
        for (const std::unique_ptr<InferencePipeline> &worker : m_workers) {
            statistics.directInferenceNanoseconds += worker->inferenceNanoseconds();
        }
        const std::uint64_t availableWorkerNanoseconds =
            m_searchWallNanoseconds * static_cast<std::uint64_t>(m_workers.size());
        statistics.directWorkerUtilization =
            availableWorkerNanoseconds == 0
                ? 0.0F
                : std::min(1.0F,
                           static_cast<float>(statistics.directInferenceNanoseconds) /
                               static_cast<float>(availableWorkerNanoseconds));
        return statistics;
    }

    void updateSearchParameters(const BatchedSearchParameters parameters) {
        m_searchParameters = parameters;
    }

    [[nodiscard]] std::vector<std::uintptr_t> workerIdentityTokens() const {
        std::vector<std::uintptr_t> identities;
        identities.reserve(m_workers.size());
        for (const std::unique_ptr<InferencePipeline> &worker : m_workers) {
            identities.push_back(reinterpret_cast<std::uintptr_t>(worker.get()));
        }
        return identities;
    }

    void refreshModel(const std::uint64_t modelGeneration, const std::string &modelPath) {
        if (modelGeneration <= m_modelGeneration) {
            throw std::invalid_argument("Model generation must increase during refresh");
        }
        if (hasPending()) {
            throw std::logic_error("Batched search must be idle during model refresh");
        }
        std::vector<PreparedInferenceModel> preparedModels;
        preparedModels.reserve(m_workers.size());
        for (const std::unique_ptr<InferencePipeline> &worker : m_workers) {
            preparedModels.push_back(worker->prepareModelRefresh(modelPath));
        }
        for (std::size_t index = 0; index < m_workers.size(); ++index) {
            m_workers[index]->commitModelRefresh(std::move(preparedModels[index]));
        }
        m_modelGeneration = modelGeneration;
        if (m_resetTreesOnRefresh) {
            for (const std::weak_ptr<Tree> &tree : m_trees) {
                if (const std::shared_ptr<Tree> active = tree.lock()) {
                    active->reset();
                }
            }
        }
    }

    [[nodiscard]] std::uint64_t modelGeneration() const noexcept { return m_modelGeneration; }

private:
    struct RootTask {
        Root root;
        std::uint32_t starting_visits;
        std::uint32_t visit_limit;
        std::uint32_t in_flight;
        bool noise_pending;
        bool count_root_initialization;
    };

    struct PendingLeaf {
        std::size_t task_index;
        std::size_t node_index;
        bool counts_as_search;
        bool root_initialization;
    };

    struct PendingBatch {
        std::size_t slot_index;
        std::vector<PendingLeaf> leaves;
    };

    BatchedInferenceParameters m_inferenceParameters;
    BatchedSearchParameters m_searchParameters;
    InferenceDimensions m_dimensions;
    std::uint64_t m_modelGeneration;
    std::vector<std::weak_ptr<Tree>> m_trees;
    std::vector<std::unique_ptr<InferencePipeline>> m_workers;
    std::vector<std::deque<PendingBatch>> m_pending;
    std::size_t m_nextWorker = 0;
    std::size_t m_nextTask = 0;
    std::mt19937 m_randomEngine;
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
    bool m_resetTreesOnRefresh;
    float m_turnDiscount;

    [[nodiscard]] std::optional<std::size_t> freeWorker() const {
        for (std::size_t offset = 0; offset < m_workers.size(); ++offset) {
            const std::size_t index = (m_nextWorker + offset) % m_workers.size();
            if (m_pending[index].size() < m_inferenceParameters.outstanding_batches_per_worker) {
                return index;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<std::size_t> readyWorker(const std::size_t first) const {
        for (std::size_t offset = 0; offset < m_workers.size(); ++offset) {
            const std::size_t index = (first + offset) % m_workers.size();
            if (!m_pending[index].empty() &&
                m_workers[index]->isCompleted(m_pending[index].front().slot_index)) {
                return index;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<std::size_t>
    schedulableTask(const std::vector<RootTask> &tasks) {
        for (std::size_t offset = 0; offset < tasks.size(); ++offset) {
            const std::size_t index = (m_nextTask + offset) % tasks.size();
            const RootTask &task = tasks[index];
            if (!task.root.tree().root().expanded() && task.in_flight != 0) {
                continue;
            }
            if (task.root.visits() < task.visit_limit &&
                task.in_flight < m_searchParameters.parallel_searches) {
                m_nextTask = (index + 1) % tasks.size();
                return index;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] bool issueBatch(std::vector<RootTask> &tasks, const std::size_t workerIndex) {
        InferencePipeline &worker = *m_workers[workerIndex];
        const InferencePipeline::WritableBatch writable = worker.acquireWritableBatch();
        std::vector<PendingLeaf> leaves;
        leaves.reserve(m_inferenceParameters.batch_size);
        try {
            while (leaves.size() < m_inferenceParameters.batch_size) {
                const std::optional<std::size_t> taskIndex = schedulableTask(tasks);
                if (!taskIndex.has_value()) {
                    break;
                }
                RootTask &task = tasks[*taskIndex];
                Tree &tree = task.root.tree();
                bool countsAsSearch = true;
                bool rootInitialization = false;
                std::optional<std::size_t> leaf;
                const auto selectionStartedAt = std::chrono::steady_clock::now();
                if (!tree.root().expanded()) {
                    if (Game::isTerminal(tree.root().position)) {
                        tree.backPropagate(tree.rootIndex(),
                                           Game::terminalValue(tree.root().position).value_or(0.0F));
                        continue;
                    }
                    leaf = tree.rootIndex();
                    rootInitialization = true;
                    countsAsSearch = task.count_root_initialization;
                    tree.node(*leaf).evaluating = true;
                } else {
                    leaf = tree.selectAvailableLeaf(m_searchParameters.exploration_constant,
                                                    m_searchParameters.minimum_root_visits);
                    if (!leaf.has_value()) {
                        break;
                    }
                    if (Game::isTerminal(tree.node(*leaf).position)) {
                        tree.backPropagate(*leaf,
                                           Game::terminalValue(tree.node(*leaf).position)
                                               .value_or(0.0F));
                        continue;
                    }
                    tree.reserve(*leaf);
                }
                m_selectionNanoseconds +=
                    static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                   std::chrono::steady_clock::now() -
                                                   selectionStartedAt)
                                                   .count());
                constexpr std::size_t encodedSize =
                    static_cast<std::size_t>(Game::inferenceDimensions().channels) *
                    static_cast<std::size_t>(Game::inferenceDimensions().rows) *
                    static_cast<std::size_t>(Game::inferenceDimensions().columns);
                const auto encodingStartedAt = std::chrono::steady_clock::now();
                Game::encodeInputInto(tree.node(*leaf).position,
                                      writable.data + leaves.size() * encodedSize);
                m_encodingNanoseconds +=
                    static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                   std::chrono::steady_clock::now() -
                                                   encodingStartedAt)
                                                   .count());
                ++task.in_flight;
                leaves.push_back({*taskIndex, *leaf, countsAsSearch, rootInitialization});
            }
        } catch (...) {
            cancelLeaves(tasks, leaves);
            worker.discardWritableBatch(writable.slotIndex);
            throw;
        }
        if (leaves.empty()) {
            worker.discardWritableBatch(writable.slotIndex);
            return false;
        }
        worker.submit(writable.slotIndex, leaves.size());
        m_pending[workerIndex].push_back({writable.slotIndex, std::move(leaves)});
        m_nextWorker = (workerIndex + 1) % m_workers.size();
        return true;
    }

    void completeWorker(std::vector<RootTask> &tasks, const std::size_t workerIndex) {
        PendingBatch pending = std::move(m_pending[workerIndex].front());
        m_pending[workerIndex].pop_front();
        InferencePipeline &worker = *m_workers[workerIndex];
        bool outputReady = false;
        std::size_t processed = 0;
        try {
            const auto waitStartedAt = std::chrono::steady_clock::now();
            const InferenceOutput output = worker.waitCompleted(pending.slot_index);
            m_waitNanoseconds +=
                static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                               std::chrono::steady_clock::now() - waitStartedAt)
                                               .count());
            outputReady = true;
            const float *policies = output.policies.data_ptr<float>();
            const float *outcomes = output.outcomes.data_ptr<float>();
            for (; processed < pending.leaves.size(); ++processed) {
                const PendingLeaf &pendingLeaf = pending.leaves[processed];
                RootTask &task = tasks[pendingLeaf.task_index];
                Tree &tree = task.root.tree();
                const auto processingStartedAt = std::chrono::steady_clock::now();
                const SearchInferenceResult<Game> inference = processSearchInference<Game>(
                    policies + processed * static_cast<std::size_t>(m_dimensions.actions),
                    outcomes + processed * static_cast<std::size_t>(m_dimensions.outcomes),
                    tree.node(pendingLeaf.node_index).position);
                m_resultProcessingNanoseconds += static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - processingStartedAt)
                        .count());
                const auto backupStartedAt = std::chrono::steady_clock::now();
                tree.expand(pendingLeaf.node_index, inference);
                if (pendingLeaf.root_initialization) {
                    tree.node(pendingLeaf.node_index).evaluating = false;
                    if (pendingLeaf.counts_as_search) {
                        tree.backPropagate(pendingLeaf.node_index, inference.value());
                    }
                    if (task.noise_pending) {
                        addNoise(task.root);
                        task.noise_pending = false;
                    }
                } else if (pendingLeaf.counts_as_search) {
                    tree.completeReservation(pendingLeaf.node_index, inference.value());
                }
                m_backupNanoseconds +=
                    static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                   std::chrono::steady_clock::now() -
                                                   backupStartedAt)
                                                   .count());
                --task.in_flight;
            }
            worker.release(pending.slot_index);
            recordBatch(pending.leaves.size());
        } catch (...) {
            if (outputReady) {
                worker.release(pending.slot_index);
            }
            std::vector<PendingLeaf> remaining(pending.leaves.begin() +
                                                   static_cast<std::ptrdiff_t>(processed),
                                               pending.leaves.end());
            cancelLeaves(tasks, remaining);
            throw;
        }
    }

    void cancelLeaves(std::vector<RootTask> &tasks,
                      const std::vector<PendingLeaf> &leaves) noexcept {
        for (const PendingLeaf &pendingLeaf : leaves) {
            RootTask &task = tasks[pendingLeaf.task_index];
            try {
                if (!pendingLeaf.root_initialization && pendingLeaf.counts_as_search) {
                    task.root.tree().cancelReservation(pendingLeaf.node_index);
                } else {
                    task.root.tree().node(pendingLeaf.node_index).evaluating = false;
                }
            } catch (...) {
            }
            --task.in_flight;
        }
    }

    void cancelPending(std::vector<RootTask> &tasks) noexcept {
        for (std::size_t workerIndex = 0; workerIndex < m_pending.size(); ++workerIndex) {
            while (!m_pending[workerIndex].empty()) {
                PendingBatch pending = std::move(m_pending[workerIndex].front());
                m_pending[workerIndex].pop_front();
                try {
                    static_cast<void>(m_workers[workerIndex]->waitCompleted(pending.slot_index));
                    m_workers[workerIndex]->release(pending.slot_index);
                } catch (...) {
                }
                cancelLeaves(tasks, pending.leaves);
            }
        }
    }

    [[nodiscard]] bool hasPending() const {
        return std::ranges::any_of(m_pending,
                                   [](const std::deque<PendingBatch> &pending) {
                                       return !pending.empty();
                                   });
    }

    void addNoise(Root &root) {
        root.tree().addRootNoise(m_searchParameters.dirichlet_alpha,
                                 m_searchParameters.dirichlet_epsilon, m_randomEngine);
    }

    void recordBatch(const std::size_t batchSize) {
        m_evaluations += batchSize;
        ++m_modelCalls;
        m_modelPositions += batchSize;
        ++m_batchHistogram[batchSize];
    }
};
