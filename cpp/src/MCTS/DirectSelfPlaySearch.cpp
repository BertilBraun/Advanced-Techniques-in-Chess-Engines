#include "DirectSelfPlaySearch.hpp"

#include "../InferenceResultProcessing.hpp"
#include "../MoveEncoding.hpp"

namespace {
constexpr std::size_t ENCODED_BOARD_SIZE = BOARD_C * BOARD_LEN * BOARD_LEN;

thread_local std::mt19937 directSelfPlayRandomEngine(std::random_device{}());

std::vector<float> dirichletNoise(const float alpha, const std::size_t count) {
    std::gamma_distribution<float> distribution(alpha, 1.0F);
    std::vector<float> noise(count);
    float sum = 0.0F;
    for (float &sample : noise) {
        sample = distribution(directSelfPlayRandomEngine);
        sum += sample;
    }
    const float normalization = 1.0F / sum;
    for (float &sample : noise) {
        sample *= normalization;
    }
    return noise;
}

MCTSResult gatherDirectResult(const MCTSRoot &root) {
    const SearchNode &rootNode = root.tree().node(root.rootIndex());
    VisitCounts visitCounts;
    visitCounts.reserve(rootNode.children.size());
    for (const Child &child : rootNode.children) {
        visitCounts.emplace_back(encodeMove(child.move, &rootNode.board),
                                 static_cast<int>(child.number_of_visits));
    }
    return {root.resultSum() / static_cast<float>(root.visits()), visitCounts, root};
}

MCTSStatistics directStatistics(const MCTSRoot &root) {
    MCTSStatistics statistics;
    statistics.averageDepth = static_cast<float>(root.maxDepth());
    const SearchNode &rootNode = root.tree().node(root.rootIndex());
    if (rootNode.children.empty()) {
        return statistics;
    }
    const float totalVisits = static_cast<float>(root.visits());
    const float uniformProbability = 1.0F / static_cast<float>(rootNode.children.size());
    for (const Child &child : rootNode.children) {
        if (child.number_of_visits == 0) {
            continue;
        }
        const float probability = static_cast<float>(child.number_of_visits) / totalVisits;
        statistics.averageEntropy -= probability * std::log2(probability);
        statistics.averageKLDivergence += probability * std::log2(probability / uniformProbability);
    }
    return statistics;
}
} // namespace

DirectSelfPlaySearch::DirectSelfPlaySearch(const InferenceClientParams &clientParameters,
                                           const MCTSParams &searchParameters,
                                           const DirectSelfPlayInferenceParams &inferenceParameters)
    : m_searchParameters(searchParameters), m_inferenceParameters(inferenceParameters),
      m_pending(static_cast<std::size_t>(inferenceParameters.inference_workers)),
      m_batchHistogram(static_cast<std::size_t>(inferenceParameters.inference_batch_size) + 1, 0) {
    m_workers.reserve(static_cast<std::size_t>(inferenceParameters.inference_workers));
    for (int worker = 0; worker < inferenceParameters.inference_workers; ++worker) {
        m_workers.push_back(std::make_unique<DirectInferencePipeline>(
            clientParameters.currentModelPath, clientParameters.device, clientParameters.device_id,
            static_cast<std::size_t>(inferenceParameters.inference_batch_size),
            static_cast<std::size_t>(
                std::max(2, inferenceParameters.outstanding_batches_per_worker)),
            true));
    }
}

std::optional<std::size_t> DirectSelfPlaySearch::freeWorker() const {
    for (std::size_t offset = 0; offset < m_workers.size(); ++offset) {
        const std::size_t workerIndex = (m_nextWorker + offset) % m_workers.size();
        if (m_pending[workerIndex].size() <
            static_cast<std::size_t>(m_inferenceParameters.outstanding_batches_per_worker)) {
            return workerIndex;
        }
    }
    return std::nullopt;
}

std::optional<std::size_t> DirectSelfPlaySearch::readyWorker(const std::size_t firstWorker) const {
    for (std::size_t offset = 0; offset < m_workers.size(); ++offset) {
        const std::size_t workerIndex = (firstWorker + offset) % m_workers.size();
        if (!m_pending[workerIndex].empty() &&
            m_workers[workerIndex]->isCompleted(m_pending[workerIndex].front().slot_index)) {
            return workerIndex;
        }
    }
    return std::nullopt;
}

std::optional<std::size_t>
DirectSelfPlaySearch::schedulableTask(const std::vector<RootTask> &tasks) {
    for (std::size_t offset = 0; offset < tasks.size(); ++offset) {
        const std::size_t taskIndex = (m_nextTask + offset) % tasks.size();
        const RootTask &task = tasks[taskIndex];
        if (!task.root.isExpanded() && task.in_flight != 0) {
            continue;
        }
        if (task.root.visits() < task.visit_limit &&
            task.in_flight < static_cast<uint32>(m_searchParameters.num_parallel_searches)) {
            m_nextTask = (taskIndex + 1) % tasks.size();
            return taskIndex;
        }
    }
    return std::nullopt;
}

std::optional<NodeIndex> DirectSelfPlaySearch::selectLeaf(MCTSRoot &root) const {
    SearchTree &tree = root.tree();
    NodeIndex selectedIndex = root.rootIndex();
    SearchNode &rootNode = tree.node(selectedIndex);
    for (uint32 childIndex = 0; childIndex < rootNode.children.size(); ++childIndex) {
        if (rootNode.children[childIndex].number_of_visits < m_searchParameters.min_visit_count) {
            selectedIndex = tree.materializeChild(selectedIndex, childIndex);
            break;
        }
    }
    while (tree.node(selectedIndex).isExpanded()) {
        const uint32 childIndex = tree.bestChildIndex(selectedIndex, m_searchParameters.c_param);
        selectedIndex = tree.materializeChild(selectedIndex, childIndex);
    }
    if (tree.node(selectedIndex).isTerminal()) {
        tree.backPropagate(selectedIndex, getBoardResultScore(tree.node(selectedIndex).board));
        return std::nullopt;
    }
    return selectedIndex;
}

void DirectSelfPlaySearch::addNoise(MCTSRoot &root) const {
    SearchNode &rootNode = root.tree().node(root.rootIndex());
    const std::vector<float> noise =
        dirichletNoise(m_searchParameters.dirichlet_alpha, rootNode.children.size());
    for (std::size_t index = 0; index < rootNode.children.size(); ++index) {
        rootNode.children[index].policy = lerp(rootNode.children[index].policy, noise[index],
                                               m_searchParameters.dirichlet_epsilon);
    }
}

void DirectSelfPlaySearch::recordBatch(const std::size_t batchSize) {
    ++m_modelCalls;
    m_modelPositions += batchSize;
    m_evaluations += batchSize;
    ++m_batchHistogram[batchSize];
}

void DirectSelfPlaySearch::completeWorker(std::vector<RootTask> &tasks,
                                          const std::size_t workerIndex,
                                          uint64 &completedSearches) {
    std::deque<PendingBatch> &pendingBatches = m_pending[workerIndex];
    if (pendingBatches.empty()) {
        throw std::logic_error("Cannot complete an idle direct self-play worker");
    }
    PendingBatch &pending = pendingBatches.front();
    DirectInferencePipeline &worker = *m_workers[workerIndex];
    std::size_t processed = 0;
    bool outputReady = false;
    try {
        const auto waitStartedAt = std::chrono::steady_clock::now();
        DirectInferenceOutput output = worker.waitCompleted(pending.slot_index);
        m_waitNanoseconds +=
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now() - waitStartedAt)
                                           .count());
        outputReady = true;
        const float *policyData = output.policies.data_ptr<float>();
        const float *outcomeData = output.outcomes.data_ptr<float>();
        for (; processed < pending.leaves.size(); ++processed) {
            const PendingLeaf &pendingLeaf = pending.leaves[processed];
            RootTask &task = tasks[pendingLeaf.task_index];
            SearchTree &tree = task.root.tree();
            SearchNode &leaf = tree.node(pendingLeaf.node_index);
            const auto processingStartedAt = std::chrono::steady_clock::now();
            const InferenceResult inferenceResult = processInferenceResult(
                policyData + processed * ACTION_SIZE, outcomeData + processed * WDL_OUTPUT_SIZE,
                leaf.board);
            m_resultProcessingNanoseconds += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - processingStartedAt)
                    .count());
            const auto backupStartedAt = std::chrono::steady_clock::now();
            tree.expand(pendingLeaf.node_index, inferenceResult.moves);
            if (pendingLeaf.counts_as_search) {
                tree.backPropagateAndRemoveVirtualLoss(pendingLeaf.node_index,
                                                       inferenceResult.value());
                ++completedSearches;
            } else if (task.noise_pending) {
                addNoise(task.root);
                task.noise_pending = false;
            }
            --task.in_flight;
            m_backupNanoseconds +=
                static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                               std::chrono::steady_clock::now() - backupStartedAt)
                                               .count());
        }
        worker.release(pending.slot_index);
        recordBatch(pending.leaves.size());
        pendingBatches.pop_front();
    } catch (...) {
        if (outputReady) {
            worker.release(pending.slot_index);
        }
        for (std::size_t index = processed; index < pending.leaves.size(); ++index) {
            const PendingLeaf &pendingLeaf = pending.leaves[index];
            RootTask &task = tasks[pendingLeaf.task_index];
            if (pendingLeaf.counts_as_search) {
                task.root.tree().removeVirtualLoss(pendingLeaf.node_index);
            }
            --task.in_flight;
        }
        pendingBatches.pop_front();
        throw;
    }
}

void DirectSelfPlaySearch::cancelPending(std::vector<RootTask> &tasks) noexcept {
    for (std::size_t workerIndex = 0; workerIndex < m_pending.size(); ++workerIndex) {
        std::deque<PendingBatch> &pendingBatches = m_pending[workerIndex];
        while (!pendingBatches.empty()) {
            PendingBatch &pending = pendingBatches.front();
            try {
                static_cast<void>(m_workers[workerIndex]->waitCompleted(pending.slot_index));
                m_workers[workerIndex]->release(pending.slot_index);
            } catch (...) {
            }
            for (const PendingLeaf &pendingLeaf : pending.leaves) {
                RootTask &task = tasks[pendingLeaf.task_index];
                if (pendingLeaf.counts_as_search) {
                    try {
                        task.root.tree().removeVirtualLoss(pendingLeaf.node_index);
                    } catch (...) {
                    }
                }
                --task.in_flight;
            }
            pendingBatches.pop_front();
        }
    }
}

MCTSResults DirectSelfPlaySearch::search(const std::vector<MCTSBoard> &boards,
                                         const bool collectStatistics) {
    if (boards.empty()) {
        return {.results = {}, .mctsStats = {}, .searchesCompleted = 0};
    }
    const auto searchStartedAt = std::chrono::steady_clock::now();
    std::vector<RootTask> tasks;
    tasks.reserve(boards.size());
    for (const MCTSBoard &board : boards) {
        MCTSRoot root = board.root;
        const uint32 visitLimit = board.should_run_full_search
                                      ? m_searchParameters.num_full_searches
                                      : m_searchParameters.num_fast_searches;
        root.tree().prepareForSearch(visitLimit,
                                     static_cast<uint32>(m_searchParameters.num_parallel_searches));
        const bool noisePending = board.should_run_full_search && !root.isExpanded();
        if (board.should_run_full_search && root.isExpanded()) {
            addNoise(root);
        }
        tasks.push_back({root, visitLimit, 0, noisePending});
    }

    uint64 completedSearches = 0;
    std::size_t completionCursor = 0;
    try {
        while (true) {
            const std::optional<std::size_t> workerIndex = freeWorker();
            if (workerIndex.has_value()) {
                DirectInferencePipeline &worker = *m_workers[*workerIndex];
                const DirectInferencePipeline::WritableBatch writable =
                    worker.acquireWritableBatch();
                std::vector<PendingLeaf> leaves;
                leaves.reserve(
                    static_cast<std::size_t>(m_inferenceParameters.inference_batch_size));
                try {
                    while (leaves.size() <
                           static_cast<std::size_t>(m_inferenceParameters.inference_batch_size)) {
                        const std::optional<std::size_t> taskIndex = schedulableTask(tasks);
                        if (!taskIndex.has_value()) {
                            break;
                        }
                        RootTask &task = tasks[*taskIndex];
                        const auto selectionStartedAt = std::chrono::steady_clock::now();
                        bool countsAsSearch = true;
                        std::optional<NodeIndex> selected;
                        if (!task.root.isExpanded()) {
                            if (task.root.isTerminal()) {
                                task.root.tree().backPropagate(
                                    task.root.rootIndex(), getBoardResultScore(task.root.board()));
                                ++completedSearches;
                                continue;
                            }
                            selected = task.root.rootIndex();
                            countsAsSearch = false;
                        } else {
                            selected = selectLeaf(task.root);
                            if (!selected.has_value()) {
                                ++completedSearches;
                                continue;
                            }
                            task.root.tree().addVirtualLoss(*selected);
                        }
                        m_selectionNanoseconds += static_cast<std::uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - selectionStartedAt)
                                .count());
                        const auto encodingStartedAt = std::chrono::steady_clock::now();
                        encodeBoardInto(task.root.tree().node(*selected).board,
                                        writable.data + leaves.size() * ENCODED_BOARD_SIZE);
                        m_encodingNanoseconds += static_cast<std::uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - encodingStartedAt)
                                .count());
                        ++task.in_flight;
                        leaves.push_back({*taskIndex, *selected, countsAsSearch});
                    }
                } catch (...) {
                    for (const PendingLeaf &pendingLeaf : leaves) {
                        RootTask &task = tasks[pendingLeaf.task_index];
                        if (pendingLeaf.counts_as_search) {
                            task.root.tree().removeVirtualLoss(pendingLeaf.node_index);
                        }
                        --task.in_flight;
                    }
                    worker.discardWritableBatch(writable.slotIndex);
                    throw;
                }
                if (leaves.empty()) {
                    worker.discardWritableBatch(writable.slotIndex);
                } else {
                    worker.submit(writable.slotIndex, leaves.size());
                    m_pending[*workerIndex].push_back({writable.slotIndex, std::move(leaves)});
                    m_nextWorker = (*workerIndex + 1) % m_workers.size();
                    continue;
                }
            }

            const bool hasPending =
                std::ranges::any_of(m_pending, [](const std::deque<PendingBatch> &batches) {
                    return !batches.empty();
                });
            if (!hasPending) {
                break;
            }
            const std::optional<std::size_t> completedWorker = readyWorker(completionCursor);
            if (completedWorker.has_value()) {
                completionCursor = *completedWorker;
            } else {
                while (m_pending[completionCursor].empty()) {
                    completionCursor = (completionCursor + 1) % m_pending.size();
                }
            }
            completeWorker(tasks, completionCursor, completedSearches);
            completionCursor = (completionCursor + 1) % m_pending.size();
        }
    } catch (...) {
        cancelPending(tasks);
        throw;
    }

    std::vector<MCTSResult> results;
    results.reserve(tasks.size());
    for (const RootTask &task : tasks) {
        assert(task.in_flight == 0);
        assert(task.root.virtualLoss() == 0.0F);
        results.push_back(gatherDirectResult(task.root));
    }
    const MCTSStatistics statistics =
        collectStatistics ? directStatistics(results.front().root) : MCTSStatistics{};
    m_searchWallNanoseconds +=
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - searchStartedAt)
                                       .count());
    return {.results = std::move(results),
            .mctsStats = statistics,
            .searchesCompleted = completedSearches};
}

InferenceResult DirectSelfPlaySearch::evaluate(const Board &board) {
    DirectInferencePipeline &worker = *m_workers.front();
    const DirectInferencePipeline::WritableBatch writable = worker.acquireWritableBatch();
    encodeBoardInto(board, writable.data);
    worker.submit(writable.slotIndex, 1);
    bool outputReady = false;
    try {
        DirectInferenceOutput output = worker.waitCompleted(writable.slotIndex);
        outputReady = true;
        const InferenceResult result = processInferenceResult(
            output.policies[0].data_ptr<float>(), output.outcomes[0].data_ptr<float>(), board);
        worker.release(writable.slotIndex);
        recordBatch(1);
        return result;
    } catch (...) {
        if (outputReady) {
            worker.release(writable.slotIndex);
        }
        throw;
    }
}

InferenceStatistics DirectSelfPlaySearch::inferenceStatistics() const {
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
