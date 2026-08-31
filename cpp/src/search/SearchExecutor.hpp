#pragma once

#include "games/GameConcepts.hpp"
#include "search/InferencePipeline.hpp"
#include "search/SearchTree.hpp"
#include "search/SearchTypes.hpp"
#include "util/Timing.hpp"
#include "util/py.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

// Runs the optimized multi-root MCTS loop, overlapping tree work with inference batches.

template <SearchGame Game> class BatchedSearchExecutor {
public:
    using Position = typename Game::State;
    using Root = GameSearchRoot<Game>;
    using Tree = GameSearchTree<Game>;

    BatchedSearchExecutor(const std::string &modelPath, const InferenceDevice device,
                          const int deviceId, const BatchedInferenceParameters inferenceParameters,
                          const BatchedSearchParameters searchParameters,
                          const InferenceExecutionOptions executionOptions)
        : m_inferenceParameters(inferenceParameters), m_searchParameters(searchParameters),
          m_pending(inferenceParameters.workers), m_randomEngine(std::random_device{}()),
          m_statistics{.modelBatchSizeHistogram =
                           std::vector<std::size_t>(inferenceParameters.batch_size + 1, 0)} {
        m_workers.resize(inferenceParameters.workers);
        for (const auto workerIndex : range(inferenceParameters.workers)) {
            m_workers[workerIndex] = std::make_unique<InferencePipeline>(
                modelPath, device, deviceId, inferenceParameters.batch_size,
                std::max<std::size_t>(2, inferenceParameters.outstanding_batches_per_worker), true,
                Game::Encoding::inferenceDimensions(), executionOptions);
        }
    }

    [[nodiscard]] GameSearchBatchResult
    searchDetailed(const std::vector<GameSearchRequest<Game>> &requests,
                   SearchBudgetAllocator *budgetAllocator = nullptr) {
        ScopedNanosecondTimer searchTimer(m_searchWallNanoseconds);
        if (requests.empty()) {
            throw std::invalid_argument("Batched search requires roots and simulations");
        }
        std::vector<RootTask> tasks;
        tasks.reserve(requests.size());
        for (const GameSearchRequest<Game> &request : requests) {
            tasks.push_back(createTask(request));
        }
        if (budgetAllocator == nullptr && std::ranges::any_of(tasks, [](const RootTask &task) {
                return !task.budget_assigned;
            })) {
            throw std::invalid_argument("Predicted search limits require a budget allocator");
        }
        std::size_t budgetCursor = 0;
        std::size_t completionCursor = 0;
        try {
            while (true) {
                assignReadyPredictedBudgets(tasks, budgetCursor, budgetAllocator);
                const std::optional<std::size_t> workerIndex = freeWorker();
                if (workerIndex.has_value() && issueBatch(tasks, *workerIndex)) {
                    continue;
                }
                if (!hasPending()) {
                    break;
                }
                completionCursor = completionWorker(completionCursor);
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
            const auto [rootPriorTopShare, rootPriorEntropy] = rootPriorFeatures(node);
            GameSearchResult result{
                .root_value = node.visits == 0 ? 0.0F : node.value_sum / node.visits,
                .highest_visited_child_action_id = -1,
                .highest_visited_child_visit_count = 0,
                .highest_visited_child_q = 0.0F,
                .search_visits = {},
                .policy_target_visits = {},
                .network_root_value = 0.0F,
                .policy_correction = 0.0F,
                .value_correction = 0.0F,
                .predicted_budget_curve = node.search_budget_curve,
                .root_prior_top_share = rootPriorTopShare,
                .root_prior_entropy = rootPriorEntropy,
                .selected_budget_index = task.selected_budget_index,
                .assigned_additional_visits = task.assigned_additional_visits,
                .parallel_searches = task.parallel_searches,
                .spend_residual = task.spend_residual,
                .starting_visits = task.starting_visits,
                .final_visits = root.visits(),
                .stop_reason = task.stop_reason,
                .checkpoints = task.checkpoints,
            };
            result.search_visits.reserve(node.children.size());
            result.policy_target_visits.reserve(node.children.size());
            std::uint32_t highestChildVisits = 0;
            const std::vector<std::uint32_t> policyTargetVisits = root.tree().policyTargetVisits(
                m_searchParameters.tree_search.exploration_constant, task.force_root_playouts);
            for (const auto index : range(node.children.size())) {
                const auto &edge = node.children[index];
                const int actionId = edge.action_id;
                if (edge.visits > 0) {
                    result.search_visits.push_back({
                        .action_id = actionId,
                        .visit_count = edge.visits,
                    });
                }
                if (policyTargetVisits[index] > 0) {
                    result.policy_target_visits.push_back({
                        .action_id = actionId,
                        .visit_count = policyTargetVisits[index],
                    });
                }
                if (edge.visits > 0 && (edge.visits > highestChildVisits ||
                                        (edge.visits == highestChildVisits &&
                                         actionId < result.highest_visited_child_action_id))) {
                    highestChildVisits = edge.visits;
                    result.highest_visited_child_action_id = actionId;
                    result.highest_visited_child_visit_count = edge.visits;
                    result.highest_visited_child_q = -root.tree().valueDiscountPerPly() *
                                                     edge.value_sum /
                                                     static_cast<float>(edge.visits);
                }
            }
            if (result.highest_visited_child_action_id < 0) {
                throw std::logic_error("Completed search has no visited child");
            }
            if (!node.network_outcome.has_value()) {
                throw std::logic_error("Completed search root has no network outcome");
            }
            result.network_root_value = node.network_outcome->expectedValue();
            result.value_correction =
                0.5F * std::abs(result.root_value - result.network_root_value);
            const std::uint64_t targetVisitTotal = std::accumulate(
                policyTargetVisits.begin(), policyTargetVisits.end(), std::uint64_t{0});
            if (targetVisitTotal == 0) {
                throw std::logic_error("Completed search has no policy-target visits");
            }
            for (const auto index : range(node.children.size())) {
                const float searchedProbability = static_cast<float>(policyTargetVisits[index]) /
                                                  static_cast<float>(targetVisitTotal);
                result.policy_correction +=
                    0.5F * std::abs(searchedProbability - node.children[index].raw_prior);
            }
            results.push_back(std::move(result));
        }
        const std::uint64_t completed =
            std::accumulate(tasks.begin(), tasks.end(), std::uint64_t{0},
                            [](const std::uint64_t count, const RootTask &task) {
                                return count + task.root.visits() - task.starting_visits;
                            });
        return {
            .results = std::move(results),
            .simulations_completed = completed,
        };
    }

    [[nodiscard]] std::vector<SearchInferenceResult<Game>>
    evaluate(const std::vector<Position> &positions) {
        std::vector<SearchInferenceResult<Game>> results;
        results.reserve(positions.size());
        std::size_t offset = 0;
        InferencePipeline &worker = *m_workers.front();
        constexpr std::size_t encodedSize = Game::Encoding::inferenceDimensions().encodedSize();
        while (offset < positions.size()) {
            const std::size_t batchSize =
                std::min(m_inferenceParameters.batch_size, positions.size() - offset);
            const InferencePipeline::WritableBatch writable = worker.acquireWritableBatch();
            for (const auto row : range(batchSize)) {
                Game::Encoding::encodeInputInto(positions[offset + row],
                                                writable.data + row * encodedSize);
            }
            worker.submit(writable.slotIndex, batchSize);
            const std::span<const Position> batchPositions(positions.data() + offset, batchSize);
            std::vector<SearchInferenceResult<Game>> batch =
                worker.consume<Game>(writable.slotIndex, batchPositions);
            results.insert(results.end(), std::make_move_iterator(batch.begin()),
                           std::make_move_iterator(batch.end()));
            recordBatch(batchSize);
            offset += batchSize;
        }
        return results;
    }

    [[nodiscard]] InferenceStatistics inferenceStatistics() const {
        InferenceStatistics statistics = m_statistics;
        statistics.averageNumberOfPositionsInInferenceCall =
            statistics.modelInferenceCalls == 0
                ? 0.0F
                : static_cast<float>(statistics.modelInferencePositions) /
                      static_cast<float>(statistics.modelInferenceCalls);
        for (const std::unique_ptr<InferencePipeline> &worker : m_workers) {
            statistics.inferenceNanoseconds += worker->inferenceNanoseconds();
            statistics.resultProcessingNanoseconds += worker->resultProcessingNanoseconds();
            statistics.treeOwnerWaitNanoseconds += worker->consumerWaitNanoseconds();
        }
        const std::uint64_t availableWorkerNanoseconds =
            m_searchWallNanoseconds * static_cast<std::uint64_t>(m_workers.size());
        statistics.workerUtilization =
            availableWorkerNanoseconds == 0
                ? 0.0F
                : std::min(1.0F, static_cast<float>(statistics.inferenceNanoseconds) /
                                     static_cast<float>(availableWorkerNanoseconds));
        return statistics;
    }

    void updateSearchParameters(const BatchedSearchParameters parameters) {
        m_searchParameters = parameters;
    }

    void refreshModel(const std::string &modelPath) {
        if (hasPending()) {
            throw std::logic_error("Batched search must be idle during model refresh");
        }
        std::vector<PreparedInferenceModel> preparedModels;
        preparedModels.reserve(m_workers.size());
        for (const std::unique_ptr<InferencePipeline> &worker : m_workers) {
            preparedModels.push_back(worker->prepareModelRefresh(modelPath));
        }
        for (const auto index : range(m_workers.size())) {
            m_workers[index]->commitModelRefresh(std::move(preparedModels[index]));
        }
    }

private:
    struct RootTask {
        Root root;
        std::uint32_t starting_visits;
        std::uint32_t root_ply;
        SearchLimit limit;
        std::uint32_t maximum_visits;
        std::uint32_t assigned_additional_visits;
        int selected_budget_index;
        std::uint32_t parallel_searches;
        std::int64_t spend_residual;
        std::size_t checkpoint_cursor;
        std::uint32_t in_flight;
        bool noise_pending;
        bool count_root_initialization;
        bool force_root_playouts;
        bool budget_assigned;
        bool selection_blocked;
        bool stopped;
        SearchStopReason stop_reason;
        SearchCheckpointDetail checkpoint_detail;
        std::vector<std::uint32_t> policy_checkpoint_visits;
        std::vector<SearchCheckpoint> checkpoints;
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

    class PendingPositions {
    public:
        PendingPositions(const std::vector<RootTask> &tasks, const std::vector<PendingLeaf> &leaves)
            : m_tasks(tasks), m_leaves(leaves) {}

        [[nodiscard]] std::size_t size() const noexcept { return m_leaves.size(); }
        [[nodiscard]] const Position &operator[](const std::size_t index) const {
            const PendingLeaf &leaf = m_leaves[index];
            return m_tasks[leaf.task_index].root.tree().position(leaf.node_index);
        }

    private:
        const std::vector<RootTask> &m_tasks;
        const std::vector<PendingLeaf> &m_leaves;
    };

    BatchedInferenceParameters m_inferenceParameters;
    BatchedSearchParameters m_searchParameters;
    std::vector<std::unique_ptr<InferencePipeline>> m_workers;
    std::vector<std::deque<PendingBatch>> m_pending;
    std::size_t m_nextWorker = 0;
    std::size_t m_nextTask = 0;
    std::mt19937 m_randomEngine;
    InferenceStatistics m_statistics;
    std::uint64_t m_searchWallNanoseconds = 0;

    [[nodiscard]] RootTask createTask(const GameSearchRequest<Game> &request) {
        if (request.root.tree().valueDiscountPerPly() !=
            m_searchParameters.tree_search.value_discount_per_ply) {
            throw std::invalid_argument(
                "Root value discount does not match the active search parameters");
        }
        if (!std::ranges::is_sorted(request.policy_checkpoint_visits) ||
            std::ranges::adjacent_find(request.policy_checkpoint_visits) !=
                request.policy_checkpoint_visits.end() ||
            std::ranges::any_of(request.policy_checkpoint_visits,
                                [](const std::uint32_t visits) { return visits == 0; })) {
            throw std::invalid_argument(
                "Policy checkpoint visits must be positive, sorted, and unique");
        }
        const std::uint32_t startingVisits = request.root.visits();
        const auto *fixed = std::get_if<FixedSearchLimit>(&request.limit);
        const auto *additional = std::get_if<AdditionalSearchLimit>(&request.limit);
        const bool predicted = std::holds_alternative<PredictedSearchBudgetLimit>(request.limit);
        if (fixed != nullptr && fixed->visits <= startingVisits) {
            throw std::invalid_argument("Fixed search limit must exceed retained root visits");
        }
        const std::uint32_t maximumAdditional = maximumAdditionalVisits(request.limit);
        const std::uint64_t maximumVisits64 =
            fixed != nullptr ? fixed->visits
                             : static_cast<std::uint64_t>(startingVisits) + maximumAdditional;
        if (maximumVisits64 > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Search visit limit exceeds the visit range");
        }
        const std::uint32_t maximumVisitLimit = static_cast<std::uint32_t>(maximumVisits64);
        if (!request.policy_checkpoint_visits.empty() &&
            (request.policy_checkpoint_visits.front() <= startingVisits ||
             request.policy_checkpoint_visits.back() > maximumVisitLimit)) {
            throw std::invalid_argument(
                "Policy checkpoints must follow retained visits and not exceed the search limit");
        }
        const std::uint32_t assignedAdditional =
            fixed != nullptr ? fixed->visits - startingVisits
                             : (additional != nullptr ? additional->additional_visits : 0U);
        const std::uint32_t parallelSearches = request.parallel_searches.value_or(
            predicted ? 16U : searchParallelism(assignedAdditional));
        if (parallelSearches == 0 || parallelSearches > 16U) {
            throw std::invalid_argument("Per-search parallelism must be between one and 16");
        }
        RootTask task{
            .root = request.root,
            .starting_visits = startingVisits,
            .root_ply = request.root_ply,
            .limit = request.limit,
            .maximum_visits = maximumVisitLimit,
            .assigned_additional_visits = assignedAdditional,
            .selected_budget_index = -1,
            .parallel_searches = parallelSearches,
            .spend_residual = 0,
            .checkpoint_cursor = 0,
            .in_flight = 0,
            .noise_pending = request.add_root_noise && !request.root.tree().root().expanded(),
            .count_root_initialization = request.count_root_initialization,
            .force_root_playouts = request.force_root_playouts,
            .budget_assigned = !predicted,
            .selection_blocked = false,
            .stopped = false,
            .stop_reason = fixed != nullptr
                               ? SearchStopReason::FixedLimit
                               : (additional != nullptr ? SearchStopReason::AdditionalVisits
                                                        : SearchStopReason::PredictedBudget),
            .checkpoint_detail = request.checkpoint_detail,
            .policy_checkpoint_visits = request.policy_checkpoint_visits,
            .checkpoints = {},
        };
        task.root.tree().prepareForSearch(task.maximum_visits, task.parallel_searches);
        if (request.add_root_noise && task.root.tree().root().expanded()) {
            addNoise(task.root);
        }
        return task;
    }

    [[nodiscard]] SearchCheckpoint checkpoint(const RootTask &task) const {
        const auto &rootNode = task.root.tree().root();
        const std::vector<std::uint32_t> policyVisits = task.root.tree().policyTargetVisits(
            m_searchParameters.tree_search.exploration_constant, task.force_root_playouts);
        const std::uint64_t total =
            std::accumulate(policyVisits.begin(), policyVisits.end(), std::uint64_t{0});
        if (total == 0) {
            throw std::logic_error("Search checkpoint has no policy visits");
        }
        std::vector<GameSearchVisit> checkpointVisits;
        if (task.checkpoint_detail == SearchCheckpointDetail::Policies) {
            checkpointVisits.reserve(policyVisits.size());
        }
        for (const auto index : range(policyVisits.size())) {
            const auto &edge = rootNode.children[index];
            if (task.checkpoint_detail == SearchCheckpointDetail::Policies &&
                policyVisits[index] > 0) {
                checkpointVisits.push_back({
                    .action_id = edge.action_id,
                    .visit_count = policyVisits[index],
                });
            }
        }
        return {
            .visits = task.root.visits(),
            .root_value = rootNode.visits == 0
                              ? 0.0F
                              : rootNode.value_sum / static_cast<float>(rootNode.visits),
            .policy_target_visits = std::move(checkpointVisits),
        };
    }

    void updateCheckpointsAndStop(RootTask &task) {
        if (!task.budget_assigned || task.in_flight != 0 || task.stopped) {
            return;
        }
        while (task.checkpoint_cursor < task.policy_checkpoint_visits.size() &&
               task.root.visits() == task.policy_checkpoint_visits[task.checkpoint_cursor]) {
            task.checkpoints.push_back(checkpoint(task));
            ++task.checkpoint_cursor;
        }
        if (task.root.visits() >= task.maximum_visits) {
            task.stopped = true;
        }
    }

    // Normalized raw-prior top share and entropy of the root: the pre-search basis a fresh root
    // exposes, recorded on every result for the analysis log.
    template <typename Node>
    [[nodiscard]] static std::pair<float, float> rootPriorFeatures(const Node &node) {
        if (node.children.empty()) {
            return {1.0F, 0.0F};
        }
        double priorTotal = 0.0;
        for (const auto &edge : node.children) {
            priorTotal += static_cast<double>(edge.raw_prior);
        }
        double topShare = 0.0;
        double entropy = 0.0;
        for (const auto &edge : node.children) {
            const double probability = priorTotal > 0.0
                                           ? static_cast<double>(edge.raw_prior) / priorTotal
                                           : 1.0 / static_cast<double>(node.children.size());
            topShare = std::max(topShare, probability);
            if (probability > 0.0) {
                entropy -= probability * std::log(probability);
            }
        }
        return {static_cast<float>(topShare), static_cast<float>(entropy)};
    }

    // Top visit share and policy entropy of the root's current policy distribution: the retained
    // visit distribution when tree reuse left one, otherwise the raw network priors. This mirrors
    // the baseline-policy features the corrector was fitted on as closely as the root allows,
    // using only information available before the search runs.
    [[nodiscard]] static SearchBudgetSelectionFeatures
    rootSelectionFeatures(const RootTask &task, const double baselineVisits,
                          const double sourceGeneration) {
        const auto &rootNode = task.root.tree().root();
        std::uint64_t totalVisits = 0;
        for (const auto &edge : rootNode.children) {
            totalVisits += edge.visits;
        }
        double topShare = 1.0;
        double entropy = 0.0;
        if (!rootNode.children.empty()) {
            topShare = 0.0;
            double priorTotal = 0.0;
            for (const auto &edge : rootNode.children) {
                priorTotal += static_cast<double>(edge.raw_prior);
            }
            for (const auto &edge : rootNode.children) {
                const double probability =
                    totalVisits > 0
                        ? static_cast<double>(edge.visits) / static_cast<double>(totalVisits)
                        : (priorTotal > 0.0 ? static_cast<double>(edge.raw_prior) / priorTotal
                                            : 1.0 / static_cast<double>(rootNode.children.size()));
                topShare = std::max(topShare, probability);
                if (probability > 0.0) {
                    entropy -= probability * std::log(probability);
                }
            }
        }
        return {
            .top_visit_share = topShare,
            .policy_entropy = entropy,
            .ply = static_cast<double>(task.root_ply),
            .baseline_visits = baselineVisits,
            .source_generation = sourceGeneration,
        };
    }

    static void assignReadyPredictedBudgets(std::vector<RootTask> &tasks, std::size_t &budgetCursor,
                                            SearchBudgetAllocator *allocator) {
        if (allocator == nullptr) {
            return;
        }
        while (budgetCursor < tasks.size()) {
            RootTask &task = tasks[budgetCursor];
            if (task.budget_assigned) {
                task.spend_residual = allocator->spendError();
                ++budgetCursor;
                continue;
            }
            if (!task.root.tree().root().expanded()) {
                return;
            }
            const auto &limit = std::get<PredictedSearchBudgetLimit>(task.limit);
            const AssignedSearchBudget assigned = allocator->assign(
                limit, task.root.tree().root().search_budget_curve,
                rootSelectionFeatures(task, static_cast<double>(limit.baseline_visits),
                                      static_cast<double>(limit.model_generation)));
            task.assigned_additional_visits = assigned.additional_visits;
            task.selected_budget_index = assigned.selected_index;
            task.spend_residual = allocator->spendError();
            const std::uint64_t finalVisits =
                static_cast<std::uint64_t>(task.starting_visits) + task.assigned_additional_visits;
            if (finalVisits > std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("Assigned search budget exceeds the visit range");
            }
            task.maximum_visits = static_cast<std::uint32_t>(finalVisits);
            if (!task.policy_checkpoint_visits.empty() &&
                task.policy_checkpoint_visits.back() > task.maximum_visits) {
                throw std::invalid_argument(
                    "Policy checkpoint exceeds the predicted search budget");
            }
            task.parallel_searches = searchParallelism(task.assigned_additional_visits);
            task.budget_assigned = true;
            ++budgetCursor;
        }
    }

    [[nodiscard]] std::optional<std::size_t> freeWorker() const {
        for (const auto offset : range(m_workers.size())) {
            const std::size_t index = (m_nextWorker + offset) % m_workers.size();
            if (m_pending[index].size() < m_inferenceParameters.outstanding_batches_per_worker) {
                return index;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<std::size_t> readyWorker(const std::size_t first) const {
        for (const auto offset : range(m_workers.size())) {
            const std::size_t index = (first + offset) % m_workers.size();
            if (!m_pending[index].empty() &&
                m_workers[index]->isCompleted(m_pending[index].front().slot_index)) {
                return index;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] std::size_t completionWorker(std::size_t first) const {
        if (const std::optional<std::size_t> ready = readyWorker(first); ready.has_value()) {
            return *ready;
        }
        while (m_pending[first].empty()) {
            first = (first + 1) % m_pending.size();
        }
        return first;
    }

    [[nodiscard]] std::optional<std::size_t> schedulableTask(const std::vector<RootTask> &tasks) {
        for (const auto offset : range(tasks.size())) {
            const std::size_t index = (m_nextTask + offset) % tasks.size();
            const RootTask &task = tasks[index];
            if (task.selection_blocked || task.stopped) {
                continue;
            }
            if (!task.root.tree().root().expanded() && task.in_flight != 0) {
                continue;
            }
            if (!task.root.tree().root().expanded()) {
                m_nextTask = (index + 1) % tasks.size();
                return index;
            }
            if (!task.budget_assigned) {
                continue;
            }
            const std::uint32_t schedulingLimit =
                task.checkpoint_cursor < task.policy_checkpoint_visits.size()
                    ? task.policy_checkpoint_visits[task.checkpoint_cursor]
                    : task.maximum_visits;
            if (task.root.visits() + task.in_flight < schedulingLimit &&
                task.in_flight < task.parallel_searches) {
                m_nextTask = (index + 1) % tasks.size();
                return index;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] bool appendInferenceLeaf(std::vector<RootTask> &tasks,
                                           const std::size_t taskIndex,
                                           const InferencePipeline::WritableBatch &writable,
                                           std::vector<PendingLeaf> &leaves) {
        RootTask &task = tasks[taskIndex];
        Tree &tree = task.root.tree();
        bool countsAsSearch = true;
        bool rootInitialization = false;
        std::optional<std::size_t> leaf;
        {
            ScopedSearchPhaseTimer selectionTimer(m_statistics.treeSelectionNanoseconds);
            if (!tree.root().expanded()) {
                if (Game::isTerminal(tree.rootPosition())) {
                    tree.backPropagate(tree.rootIndex(), Game::terminalValue(tree.rootPosition()));
                    return true;
                }
                leaf = tree.rootIndex();
                rootInitialization = true;
                countsAsSearch = task.count_root_initialization;
                tree.node(*leaf).inference_pending = true;
            } else {
                leaf = tree.selectAvailableLeaf(m_searchParameters.tree_search,
                                                task.force_root_playouts);
                if (!leaf.has_value()) {
                    task.selection_blocked = true;
                    return false;
                }
                if (Game::isTerminal(tree.position(*leaf))) {
                    tree.backPropagate(*leaf, Game::terminalValue(tree.position(*leaf)));
                    updateCheckpointsAndStop(task);
                    return true;
                }
                tree.reserve(*leaf);
            }
        }
        constexpr std::size_t encodedSize = Game::Encoding::inferenceDimensions().encodedSize();
        {
            ScopedSearchPhaseTimer encodingTimer(m_statistics.boardEncodingNanoseconds);
            Game::Encoding::encodeInputInto(tree.position(*leaf),
                                            writable.data + leaves.size() * encodedSize);
        }
        ++task.in_flight;
        leaves.push_back({
            .task_index = taskIndex,
            .node_index = *leaf,
            .counts_as_search = countsAsSearch,
            .root_initialization = rootInitialization,
        });
        return true;
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
                if (!appendInferenceLeaf(tasks, *taskIndex, writable, leaves)) {
                    // Only a completing reservation can free a leaf in that tree, so the task is
                    // skipped rather than ending the fill: one blocked tree used to truncate the
                    // whole inference batch.
                    continue;
                }
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
        m_pending[workerIndex].push_back({
            .slot_index = writable.slotIndex,
            .leaves = std::move(leaves),
        });
        m_nextWorker = (workerIndex + 1) % m_workers.size();
        return true;
    }

    void completeLeaf(RootTask &task, const PendingLeaf &pendingLeaf,
                      const SearchInferenceResult<Game> &inference) {
        Tree &tree = task.root.tree();
        ScopedSearchPhaseTimer backupTimer(m_statistics.treeBackupNanoseconds);
        tree.expand(pendingLeaf.node_index, inference);
        if (pendingLeaf.root_initialization) {
            tree.node(pendingLeaf.node_index).inference_pending = false;
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
        --task.in_flight;
        task.selection_blocked = false;
        updateCheckpointsAndStop(task);
    }

    void completeWorker(std::vector<RootTask> &tasks, const std::size_t workerIndex) {
        PendingBatch pending = std::move(m_pending[workerIndex].front());
        m_pending[workerIndex].pop_front();
        InferencePipeline &worker = *m_workers[workerIndex];
        std::size_t processed = 0;
        try {
            const PendingPositions positions(tasks, pending.leaves);
            std::vector<SearchInferenceResult<Game>> inferenceResults =
                worker.consume<Game>(pending.slot_index, positions);
            for (const auto leafIndex : range(pending.leaves.size())) {
                processed = leafIndex;
                const PendingLeaf &pendingLeaf = pending.leaves[processed];
                RootTask &task = tasks[pendingLeaf.task_index];
                completeLeaf(task, pendingLeaf, inferenceResults[leafIndex]);
                ++processed;
            }
            recordBatch(pending.leaves.size());
        } catch (...) {
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
                    task.root.tree().node(pendingLeaf.node_index).inference_pending = false;
                }
            } catch (...) {
            }
            --task.in_flight;
            task.selection_blocked = false;
        }
    }

    void cancelPending(std::vector<RootTask> &tasks) noexcept {
        for (const auto workerIndex : range(m_pending.size())) {
            while (!m_pending[workerIndex].empty()) {
                PendingBatch pending = std::move(m_pending[workerIndex].front());
                m_pending[workerIndex].pop_front();
                try {
                    m_workers[workerIndex]->consumeWithoutResult(pending.slot_index);
                } catch (...) {
                }
                cancelLeaves(tasks, pending.leaves);
            }
        }
    }

    [[nodiscard]] bool hasPending() const {
        return std::ranges::any_of(
            m_pending, [](const std::deque<PendingBatch> &pending) { return !pending.empty(); });
    }

    void addNoise(Root &root) {
        root.tree().addRootNoise(m_searchParameters.dirichlet_alpha,
                                 m_searchParameters.dirichlet_epsilon, m_randomEngine);
    }

    void recordBatch(const std::size_t batchSize) {
        m_statistics.evaluations += batchSize;
        ++m_statistics.modelInferenceCalls;
        m_statistics.modelInferencePositions += batchSize;
        ++m_statistics.modelBatchSizeHistogram[batchSize];
    }
};
