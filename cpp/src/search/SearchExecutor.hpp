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
                          const SdpaBackend sdpaBackend)
        : m_inferenceParameters(inferenceParameters), m_searchParameters(searchParameters),
          m_pending(inferenceParameters.workers), m_randomEngine(std::random_device{}()),
          m_statistics{.modelBatchSizeHistogram =
                           std::vector<std::size_t>(inferenceParameters.batch_size + 1, 0)} {
        m_workers.resize(inferenceParameters.workers);
        for (const auto workerIndex : range(inferenceParameters.workers)) {
            m_workers[workerIndex] = std::make_unique<InferencePipeline>(
                modelPath, device, deviceId, inferenceParameters.batch_size,
                std::max<std::size_t>(2, inferenceParameters.outstanding_batches_per_worker), true,
                Game::Encoding::inferenceDimensions(), sdpaBackend);
        }
    }

    [[nodiscard]] GameSearchBatchResult
    searchDetailed(const std::vector<GameSearchRequest<Game>> &requests) {
        ScopedNanosecondTimer searchTimer(m_searchWallNanoseconds);
        const bool hasInvalidRequest =
            std::ranges::any_of(requests, [](const GameSearchRequest<Game> &request) {
                return maximumVisits(request.limit) == 0;
            });
        if (requests.empty() || hasInvalidRequest) {
            throw std::invalid_argument("Batched search requires roots and simulations");
        }
        std::vector<RootTask> tasks;
        tasks.reserve(requests.size());
        for (const GameSearchRequest<Game> &request : requests) {
            tasks.push_back(createTask(request));
        }
        std::size_t completionCursor = 0;
        try {
            while (true) {
                const bool admittedWaitingSearch = admitWaitingFastSearches(tasks);
                const std::optional<std::size_t> workerIndex = freeWorker();
                if (workerIndex.has_value() && issueBatch(tasks, *workerIndex)) {
                    continue;
                }
                if (admitWaitingFastSearches(tasks) || admittedWaitingSearch) {
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
                .search_correction_target = 0.0F,
                .predicted_search_correction = node.search_correction,
                .starting_visits = task.starting_visits,
                .final_visits = root.visits(),
                .stop_reason = task.stop_reason,
                .learned_gate_evaluated = task.learned_gate_checked,
                .checkpoints = task.checkpoints,
            };
            result.search_visits.reserve(node.children.size());
            result.policy_target_visits.reserve(node.children.size());
            std::uint32_t highestChildVisits = 0;
            const std::vector<std::uint32_t> policyTargetVisits = root.tree().policyTargetVisits(
                m_searchParameters.tree_search.exploration_constant, task.force_root_playouts);
            for (const auto index : range(node.children.size())) {
                const auto &edge = node.children[index];
                const int actionId = Game::Encoding::actionId(edge.action, node.position);
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
            const std::uint64_t targetVisitTotal =
                std::accumulate(policyTargetVisits.begin(), policyTargetVisits.end(),
                                std::uint64_t{0});
            if (targetVisitTotal == 0) {
                throw std::logic_error("Completed search has no policy-target visits");
            }
            for (const auto index : range(node.children.size())) {
                const float searchedProbability =
                    static_cast<float>(policyTargetVisits[index]) /
                    static_cast<float>(targetVisitTotal);
                result.policy_correction +=
                    0.5F * std::abs(searchedProbability - node.children[index].raw_prior);
            }
            result.search_correction_target =
                std::max(result.policy_correction, result.value_correction);
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
        SearchLimit limit;
        std::uint32_t maximum_visits;
        std::uint32_t next_observation_visits;
        std::uint32_t in_flight;
        bool noise_pending;
        bool count_root_initialization;
        bool force_root_playouts;
        bool admitted;
        bool staged_fast_search;
        bool completion_recorded;
        bool stopped;
        bool learned_gate_checked;
        SearchStopReason stop_reason;
        SearchCheckpointDetail checkpoint_detail;
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
            return m_tasks[leaf.task_index].root.tree().node(leaf.node_index).position;
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
        RootTask task{
            .root = request.root,
            .starting_visits = request.root.visits(),
            .limit = request.limit,
            .maximum_visits = maximumVisits(request.limit),
            .next_observation_visits = nextObservationVisits(request),
            .in_flight = 0,
            .noise_pending = request.add_root_noise && !request.root.tree().root().expanded(),
            .count_root_initialization = request.count_root_initialization,
            .force_root_playouts = request.force_root_playouts,
            .admitted = request.admission != SearchAdmission::WaitingFast,
            .staged_fast_search = request.admission != SearchAdmission::Immediate,
            .completion_recorded = false,
            .stopped = request.root.visits() >= maximumVisits(request.limit),
            .learned_gate_checked = false,
            .stop_reason = std::holds_alternative<FixedSearchLimit>(request.limit)
                               ? SearchStopReason::FixedLimit
                               : SearchStopReason::Maximum,
            .checkpoint_detail = request.checkpoint_detail,
            .checkpoints = {},
        };
        task.root.tree().prepareForSearch(task.maximum_visits,
                                          m_searchParameters.parallel_searches);
        if (request.add_root_noise && task.root.tree().root().expanded()) {
            addNoise(task.root);
        }
        return task;
    }

    [[nodiscard]] static std::uint32_t
    nextObservationVisits(const GameSearchRequest<Game> &request) {
        const auto *adaptive = std::get_if<AdaptiveSearchLimit>(&request.limit);
        if (adaptive == nullptr) {
            return maximumVisits(request.limit) + 1U;
        }
        const std::uint32_t completed = request.root.visits();
        return (completed / adaptive->observation_interval + 1U) *
               adaptive->observation_interval;
    }

    [[nodiscard]] SearchCheckpoint checkpoint(const RootTask &task,
                                              const AdaptiveSearchLimit &limit) const {
        const auto &rootNode = task.root.tree().root();
        const std::vector<std::uint32_t> policyVisits = task.root.tree().policyTargetVisits(
            m_searchParameters.tree_search.exploration_constant, task.force_root_playouts);
        const std::uint64_t total =
            std::accumulate(policyVisits.begin(), policyVisits.end(), std::uint64_t{0});
        if (total == 0) {
            throw std::logic_error("Adaptive search checkpoint has no policy visits");
        }
        std::size_t leader = 0;
        std::size_t runnerUp = 0;
        for (const auto index : range(policyVisits.size())) {
            if (policyVisits[index] > policyVisits[leader]) {
                runnerUp = leader;
                leader = index;
            } else if (index != leader &&
                       (runnerUp == leader || policyVisits[index] > policyVisits[runnerUp])) {
                runnerUp = index;
            }
        }
        const float rootValue = rootNode.value_sum / static_cast<float>(rootNode.visits);
        const float topShare =
            static_cast<float>(policyVisits[leader]) / static_cast<float>(total);
        const float secondShare = policyVisits.size() == 1
                                      ? 0.0F
                                      : static_cast<float>(policyVisits[runnerUp]) /
                                            static_cast<float>(total);
        const std::size_t historyCount =
            limit.leader_stability_window / limit.observation_interval;
        bool stable = task.checkpoints.size() >= historyCount;
        const int leaderAction =
            Game::Encoding::actionId(rootNode.children[leader].action, rootNode.position);
        int mostVisitedAction = -1;
        std::uint32_t mostVisits = 0;
        std::vector<GameSearchVisit> checkpointVisits;
        if (task.checkpoint_detail == SearchCheckpointDetail::Policies) {
            checkpointVisits.reserve(policyVisits.size());
        }
        for (const auto index : range(policyVisits.size())) {
            const auto &edge = rootNode.children[index];
            const int actionId = Game::Encoding::actionId(edge.action, rootNode.position);
            if (mostVisitedAction < 0 || edge.visits > mostVisits ||
                (edge.visits == mostVisits && actionId < mostVisitedAction)) {
                mostVisitedAction = actionId;
                mostVisits = edge.visits;
            }
            if (task.checkpoint_detail == SearchCheckpointDetail::Policies &&
                policyVisits[index] > 0) {
                checkpointVisits.push_back({
                    .action_id = actionId,
                    .visit_count = policyVisits[index],
                });
            }
        }
        if (mostVisitedAction < 0) {
            throw std::logic_error("Adaptive search checkpoint has no visited child");
        }
        if (stable) {
            for (const SearchCheckpoint &previous :
                 std::span(task.checkpoints).last(historyCount)) {
                stable = stable && previous.leader_action_id == leaderAction;
            }
        }
        const float valueDelta = task.checkpoints.empty()
                                     ? 2.0F
                                     : std::abs(rootValue - task.checkpoints.back().root_value);
        return {
            .visits = task.root.visits(),
            .leader_action_id = leaderAction,
            .most_visited_action_id = mostVisitedAction,
            .policy_target_visits = std::move(checkpointVisits),
            .top_visit_share = topShare,
            .top_two_margin = topShare - secondShare,
            .root_value = rootValue,
            .root_value_delta = valueDelta,
            .leader_stable = stable,
        };
    }

    static bool deterministicStop(const SearchCheckpoint &current,
                                  const AdaptiveSearchLimit &limit) {
        if (current.visits < limit.minimum_visits || !current.leader_stable ||
            current.root_value_delta > limit.root_value_tolerance) {
            return false;
        }
        const float progress = std::clamp(
            static_cast<float>(current.visits - limit.minimum_visits) /
                static_cast<float>(limit.threshold_relaxation_visits),
            0.0F, 1.0F);
        const float topShare =
            std::lerp(limit.initial_top_visit_share, limit.final_top_visit_share, progress);
        const float margin =
            std::lerp(limit.initial_top_two_margin, limit.final_top_two_margin, progress);
        return current.top_visit_share >= topShare || current.top_two_margin >= margin;
    }

    void updateAdaptiveStop(RootTask &task) {
        const auto *limit = std::get_if<AdaptiveSearchLimit>(&task.limit);
        if (limit == nullptr || task.stopped || task.root.visits() < task.next_observation_visits) {
            return;
        }
        const SearchCheckpoint current = checkpoint(task, *limit);
        task.checkpoints.push_back(current);
        task.next_observation_visits += limit->observation_interval;
        if (deterministicStop(current, *limit)) {
            task.stopped = true;
            task.stop_reason = SearchStopReason::Deterministic;
            return;
        }
        if (const auto *gate =
                std::get_if<SearchCorrectionGate>(&limit->search_correction_gate);
            gate != nullptr && !task.learned_gate_checked && current.visits >= gate->visit) {
            task.learned_gate_checked = true;
            if (task.root.tree().root().search_correction < gate->minimum_prediction) {
                task.stopped = true;
                task.stop_reason = SearchStopReason::LearnedGate;
                return;
            }
        }
        if (task.root.visits() >= task.maximum_visits) {
            task.stopped = true;
            task.stop_reason = SearchStopReason::Maximum;
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
            if (!task.admitted) {
                continue;
            }
            if (!task.root.tree().root().expanded() && task.in_flight != 0) {
                continue;
            }
            const std::uint32_t schedulingLimit =
                std::min(task.maximum_visits, task.next_observation_visits);
            if (!task.stopped && task.root.visits() < schedulingLimit &&
                task.in_flight < m_searchParameters.parallel_searches) {
                m_nextTask = (index + 1) % tasks.size();
                return index;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] bool admitWaitingFastSearches(std::vector<RootTask> &tasks) {
        bool admitted = false;
        for (RootTask &completedTask : tasks) {
            if (!completedTask.staged_fast_search || !completedTask.admitted ||
                completedTask.completion_recorded || completedTask.in_flight != 0 ||
                (!completedTask.stopped &&
                 completedTask.root.visits() < completedTask.maximum_visits)) {
                continue;
            }
            completedTask.completion_recorded = true;
            const auto waiting =
                std::ranges::find_if(tasks, [](const RootTask &task) { return !task.admitted; });
            if (waiting != tasks.end()) {
                waiting->admitted = true;
                admitted = true;
            }
        }
        return admitted;
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
            ScopedNanosecondTimer selectionTimer(m_statistics.treeSelectionNanoseconds);
            if (!tree.root().expanded()) {
                if (Game::isTerminal(tree.root().position)) {
                    tree.backPropagate(tree.rootIndex(), Game::terminalValue(tree.root().position));
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
                    return false;
                }
                if (Game::isTerminal(tree.node(*leaf).position)) {
                    tree.backPropagate(*leaf, Game::terminalValue(tree.node(*leaf).position));
                    if (task.in_flight == 0) {
                        updateAdaptiveStop(task);
                    }
                    return true;
                }
                tree.reserve(*leaf);
            }
        }
        constexpr std::size_t encodedSize = Game::Encoding::inferenceDimensions().encodedSize();
        {
            ScopedNanosecondTimer encodingTimer(m_statistics.boardEncodingNanoseconds);
            Game::Encoding::encodeInputInto(tree.node(*leaf).position,
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
                    break;
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
        ScopedNanosecondTimer backupTimer(m_statistics.treeBackupNanoseconds);
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
        if (task.in_flight == 0) {
            updateAdaptiveStop(task);
        }
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
