#pragma once

#include "DirectInference.hpp"
#include "SearchInference.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

template <typename Action> struct GameSearchEdge {
    Action action;
    float raw_prior;
    float prior;
    std::uint32_t visits = 0;
    float value_sum = 0.0F;
    float virtual_loss = 0.0F;
    std::optional<std::size_t> child_index;
};

template <typename Game> struct GameSearchNode {
    using Position = typename Game::Position;
    using Action = typename Game::Action;

    Position position;
    std::vector<GameSearchEdge<Action>> children;
    std::optional<std::size_t> parent_index;
    std::optional<std::size_t> parent_edge_index;
    std::uint32_t visits = 0;
    float value_sum = 0.0F;
    float virtual_loss = 0.0F;
    bool evaluating = false;

    [[nodiscard]] bool expanded() const noexcept { return !children.empty(); }
};

template <typename Game> class GameSearchTree {
public:
    using Position = typename Game::Position;
    using Action = typename Game::Action;
    using Node = GameSearchNode<Game>;
    using Edge = GameSearchEdge<Action>;

    GameSearchTree(Position rootPosition, const std::size_t capacity)
        : m_capacity(capacity), m_initialPosition(rootPosition) {
        if (capacity == 0) {
            throw std::invalid_argument("Game search tree capacity must be positive");
        }
        m_nodes.reserve(capacity);
        m_nodes.push_back({std::move(rootPosition), {}, std::nullopt, std::nullopt});
    }

    [[nodiscard]] const Node &node(const std::size_t index) const { return m_nodes.at(index); }
    [[nodiscard]] Node &node(const std::size_t index) { return m_nodes.at(index); }
    [[nodiscard]] const Node &root() const { return node(m_rootIndex); }
    [[nodiscard]] Node &root() { return node(m_rootIndex); }
    [[nodiscard]] std::size_t rootIndex() const noexcept { return m_rootIndex; }
    [[nodiscard]] std::size_t liveNodeCount() const noexcept { return m_nodes.size(); }
    [[nodiscard]] std::size_t capacity() const noexcept { return m_capacity; }

    [[nodiscard]] std::optional<std::size_t>
    selectAvailableLeaf(const float explorationConstant, const std::uint32_t minimumRootVisits = 0) {
        Node &rootNode = root();
        for (std::size_t edgeIndex = 0; edgeIndex < rootNode.children.size(); ++edgeIndex) {
            if (rootNode.children[edgeIndex].visits >= minimumRootVisits) {
                continue;
            }
            const std::optional<std::size_t> leaf = selectAvailableLeaf(
                materializeChild(m_rootIndex, edgeIndex), explorationConstant);
            if (leaf.has_value()) {
                return leaf;
            }
        }
        return selectAvailableLeaf(m_rootIndex, explorationConstant);
    }

    void expand(const std::size_t nodeIndex,
                const SearchInferenceResult<Game> &inferenceResult) {
        Node &selected = node(nodeIndex);
        if (selected.expanded() || Game::isTerminal(selected.position)) {
            return;
        }
        selected.children.reserve(inferenceResult.actions.size());
        for (const auto &[action, prior] : inferenceResult.actions) {
            selected.children.push_back({action, prior, prior});
        }
    }

    void backPropagate(std::size_t nodeIndex, float value) {
        while (true) {
            Node &selected = node(nodeIndex);
            ++selected.visits;
            selected.value_sum += value;
            if (!selected.parent_index.has_value()) {
                break;
            }
            Node &parent = node(*selected.parent_index);
            Edge &incoming = parent.children.at(*selected.parent_edge_index);
            ++incoming.visits;
            incoming.value_sum += value;
            value = -value;
            nodeIndex = *selected.parent_index;
        }
    }

    void reserve(const std::size_t nodeIndex) {
        if (node(nodeIndex).evaluating) {
            throw std::logic_error("Game search leaf is already reserved");
        }
        node(nodeIndex).evaluating = true;
        updatePath(nodeIndex, 1, 0.0F, 1.0F);
    }

    void cancelReservation(const std::size_t nodeIndex) {
        if (!node(nodeIndex).evaluating) {
            throw std::logic_error("Game search leaf is not reserved");
        }
        updatePath(nodeIndex, -1, 0.0F, -1.0F);
        node(nodeIndex).evaluating = false;
    }

    void completeReservation(const std::size_t nodeIndex, const float value) {
        if (!node(nodeIndex).evaluating) {
            throw std::logic_error("Game search leaf is not reserved");
        }
        completeReservedPath(nodeIndex, value);
        node(nodeIndex).evaluating = false;
    }

    void addRootNoise(const float alpha, const float epsilon, std::mt19937 &randomEngine) {
        Node &rootNode = root();
        if (rootNode.children.empty()) {
            return;
        }
        std::gamma_distribution<float> distribution(alpha, 1.0F);
        std::vector<float> noise(rootNode.children.size());
        float sum = 0.0F;
        for (float &sample : noise) {
            sample = distribution(randomEngine);
            sum += sample;
        }
        for (std::size_t index = 0; index < rootNode.children.size(); ++index) {
            rootNode.children[index].prior =
                std::lerp(rootNode.children[index].prior, noise[index] / sum, epsilon);
        }
    }

    [[nodiscard]] std::size_t actionIdToEdgeIndex(const int actionId) const {
        const Node &selected = root();
        for (std::size_t index = 0; index < selected.children.size(); ++index) {
            if (Game::actionId(selected.children[index].action, selected.position) == actionId) {
                return index;
            }
        }
        throw std::invalid_argument("Selected action is not a root child");
    }

    void reroot(const int actionId) {
        const std::size_t edgeIndex = actionIdToEdgeIndex(actionId);
        const Position nextPosition =
            Game::childPosition(root().position, root().children[edgeIndex].action);
        m_initialPosition = nextPosition;
        reset();
    }

    void reset() {
        m_nodes.clear();
        m_rootIndex = 0;
        m_nodes.push_back({m_initialPosition, {}, std::nullopt, std::nullopt});
    }

private:
    std::size_t m_capacity;
    Position m_initialPosition;
    std::vector<Node> m_nodes;
    std::size_t m_rootIndex = 0;

    [[nodiscard]] std::size_t bestEdgeIndex(const std::size_t nodeIndex,
                                            const float explorationConstant) const {
        const Node &parent = node(nodeIndex);
        const float parentScale = std::sqrt(static_cast<float>(std::max(1U, parent.visits)));
        float bestScore = -std::numeric_limits<float>::infinity();
        std::size_t bestIndex = 0;
        for (std::size_t index = 0; index < parent.children.size(); ++index) {
            const Edge &edge = parent.children[index];
            const float meanValue =
                edge.visits == 0 ? 0.0F : (edge.value_sum + edge.virtual_loss) / edge.visits;
            const float exploration =
                explorationConstant * edge.prior * parentScale / (1.0F + edge.visits);
            const float score = -meanValue + exploration;
            if (score > bestScore) {
                bestScore = score;
                bestIndex = index;
            }
        }
        return bestIndex;
    }

    [[nodiscard]] std::optional<std::size_t>
    selectAvailableLeaf(const std::size_t nodeIndex, const float explorationConstant) {
        const Node &selected = node(nodeIndex);
        if (!selected.expanded()) {
            return selected.evaluating ? std::nullopt
                                       : std::optional<std::size_t>(nodeIndex);
        }
        std::vector<bool> attempted(selected.children.size(), false);
        for (std::size_t attempt = 0; attempt < selected.children.size(); ++attempt) {
            float bestScore = -std::numeric_limits<float>::infinity();
            std::size_t bestIndex = 0;
            const float parentScale =
                std::sqrt(static_cast<float>(std::max(1U, selected.visits)));
            for (std::size_t index = 0; index < selected.children.size(); ++index) {
                if (attempted[index]) {
                    continue;
                }
                const Edge &edge = selected.children[index];
                const float meanValue = edge.visits == 0
                                            ? 0.0F
                                            : (edge.value_sum + edge.virtual_loss) / edge.visits;
                const float score = -meanValue + explorationConstant * edge.prior * parentScale /
                                                     (1.0F + edge.visits);
                if (score > bestScore) {
                    bestScore = score;
                    bestIndex = index;
                }
            }
            attempted[bestIndex] = true;
            const std::optional<std::size_t> leaf =
                selectAvailableLeaf(materializeChild(nodeIndex, bestIndex), explorationConstant);
            if (leaf.has_value()) {
                return leaf;
            }
        }
        return std::nullopt;
    }

    void updatePath(std::size_t nodeIndex, const int visitDelta, float value,
                    const float virtualLossDelta) {
        while (true) {
            Node &selected = node(nodeIndex);
            selected.visits = static_cast<std::uint32_t>(
                static_cast<std::int64_t>(selected.visits) + visitDelta);
            selected.value_sum += value;
            selected.virtual_loss += virtualLossDelta;
            if (!selected.parent_index.has_value()) {
                break;
            }
            Edge &incoming =
                node(*selected.parent_index).children.at(*selected.parent_edge_index);
            incoming.visits = static_cast<std::uint32_t>(
                static_cast<std::int64_t>(incoming.visits) + visitDelta);
            incoming.value_sum += value;
            incoming.virtual_loss += virtualLossDelta;
            value = -value;
            nodeIndex = *selected.parent_index;
        }
    }

    void completeReservedPath(std::size_t nodeIndex, float value) {
        while (true) {
            Node &selected = node(nodeIndex);
            selected.value_sum += value;
            selected.virtual_loss -= 1.0F;
            if (!selected.parent_index.has_value()) {
                break;
            }
            Edge &incoming =
                node(*selected.parent_index).children.at(*selected.parent_edge_index);
            incoming.value_sum += value;
            incoming.virtual_loss -= 1.0F;
            value = -value;
            nodeIndex = *selected.parent_index;
        }
    }

    [[nodiscard]] std::size_t materializeChild(const std::size_t parentIndex,
                                               const std::size_t edgeIndex) {
        Edge &edge = node(parentIndex).children.at(edgeIndex);
        if (edge.child_index.has_value()) {
            return *edge.child_index;
        }
        if (m_nodes.size() == m_capacity) {
            throw std::overflow_error("Game search tree capacity exhausted");
        }
        const Position child = Game::childPosition(node(parentIndex).position, edge.action);
        const std::size_t childIndex = m_nodes.size();
        m_nodes.push_back({child, {}, parentIndex, edgeIndex});
        node(parentIndex).children[edgeIndex].child_index = childIndex;
        return childIndex;
    }
};

template <typename Game> class GameSearchRoot {
public:
    using Position = typename Game::Position;
    using Tree = GameSearchTree<Game>;

    GameSearchRoot(Position position, const std::size_t capacity)
        : m_tree(std::make_shared<Tree>(std::move(position), capacity)) {}

    [[nodiscard]] const Position &position() const { return m_tree->root().position; }
    [[nodiscard]] bool isTerminal() const { return Game::isTerminal(position()); }
    [[nodiscard]] std::uint32_t visits() const { return m_tree->root().visits; }
    [[nodiscard]] std::size_t liveNodeCount() const { return m_tree->liveNodeCount(); }
    [[nodiscard]] Tree &tree() { return *m_tree; }
    [[nodiscard]] const Tree &tree() const { return *m_tree; }
    [[nodiscard]] std::shared_ptr<Tree> sharedTree() const { return m_tree; }
    void play(const int actionId) { m_tree->reroot(actionId); }
    void reset() { m_tree->reset(); }

private:
    std::shared_ptr<Tree> m_tree;
};

struct GameSearchVisit {
    int action_id;
    std::uint32_t visit_count;
};

struct GameSearchResult {
    float root_value;
    std::vector<GameSearchVisit> visits;
};

struct BatchedSearchParameters {
    std::uint32_t parallel_searches;
    float exploration_constant;
    std::uint32_t minimum_root_visits;
    float dirichlet_alpha;
    float dirichlet_epsilon;
    std::size_t tree_capacity;

    BatchedSearchParameters(std::uint32_t parallelSearches, float explorationConstant,
                            std::uint32_t minimumRootVisits, float dirichletAlpha,
                            float dirichletEpsilon, std::size_t treeCapacity)
        : parallel_searches(parallelSearches), exploration_constant(explorationConstant),
          minimum_root_visits(minimumRootVisits), dirichlet_alpha(dirichletAlpha),
          dirichlet_epsilon(dirichletEpsilon), tree_capacity(treeCapacity) {
        if (parallel_searches == 0 || tree_capacity == 0) {
            throw std::invalid_argument("Batched search counts and tree capacity must be positive");
        }
        if (exploration_constant <= 0.0F || dirichlet_alpha <= 0.0F ||
            dirichlet_epsilon < 0.0F || dirichlet_epsilon > 1.0F) {
            throw std::invalid_argument("Batched search constants are outside their valid range");
        }
    }
};

struct BatchedInferenceParameters {
    std::size_t workers;
    std::size_t batch_size;
    std::size_t outstanding_batches_per_worker;

    BatchedInferenceParameters(std::size_t inferenceWorkers, std::size_t inferenceBatchSize,
                               std::size_t outstandingBatchesPerWorker)
        : workers(inferenceWorkers), batch_size(inferenceBatchSize),
          outstanding_batches_per_worker(outstandingBatchesPerWorker) {
        if (workers == 0 || batch_size == 0 || outstanding_batches_per_worker == 0 ||
            outstanding_batches_per_worker > 2) {
            throw std::invalid_argument(
                "Batched inference counts must be positive and outstanding batches at most two");
        }
    }
};

template <typename Game> class BatchedGameSearch {
public:
    using Root = GameSearchRoot<Game>;
    using Tree = GameSearchTree<Game>;

    BatchedGameSearch(const std::string &modelPath, const InferenceDevice device,
                      const int deviceId, const BatchedInferenceParameters inferenceParameters,
                      const BatchedSearchParameters searchParameters,
                      const std::uint64_t modelGeneration)
        : m_inferenceParameters(inferenceParameters), m_searchParameters(searchParameters),
          m_dimensions(Game::inferenceDimensions()), m_modelGeneration(modelGeneration),
          m_pending(inferenceParameters.workers), m_randomEngine(std::random_device{}()) {
        m_workers.reserve(inferenceParameters.workers);
        for (std::size_t worker = 0; worker < inferenceParameters.workers; ++worker) {
            m_workers.push_back(std::make_unique<DirectInferencePipeline>(
                modelPath, device, deviceId, inferenceParameters.batch_size,
                std::max<std::size_t>(2, inferenceParameters.outstanding_batches_per_worker),
                true, m_dimensions));
        }
    }

    [[nodiscard]] Root newRoot(typename Game::Position position) {
        Root root(std::move(position), m_searchParameters.tree_capacity);
        m_trees.push_back(root.sharedTree());
        return root;
    }

    [[nodiscard]] std::vector<GameSearchResult> search(std::vector<Root> &roots,
                                                       const std::uint32_t simulations) {
        if (roots.empty() || simulations == 0) {
            throw std::invalid_argument("Batched search requires roots and simulations");
        }
        std::vector<RootTask> tasks;
        tasks.reserve(roots.size());
        for (Root &root : roots) {
            tasks.push_back({root, root.visits() + simulations, 0, !root.tree().root().expanded()});
            if (root.tree().root().expanded()) {
                addNoise(root);
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
        return results;
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
        for (const std::unique_ptr<DirectInferencePipeline> &worker : m_workers) {
            preparedModels.push_back(worker->prepareModelRefresh(modelPath));
        }
        for (std::size_t index = 0; index < m_workers.size(); ++index) {
            m_workers[index]->commitModelRefresh(std::move(preparedModels[index]));
        }
        m_modelGeneration = modelGeneration;
        for (const std::weak_ptr<Tree> &tree : m_trees) {
            if (const std::shared_ptr<Tree> active = tree.lock()) {
                active->reset();
            }
        }
    }

    [[nodiscard]] std::uint64_t modelGeneration() const noexcept { return m_modelGeneration; }

private:
    struct RootTask {
        Root root;
        std::uint32_t visit_limit;
        std::uint32_t in_flight;
        bool noise_pending;
    };

    struct PendingLeaf {
        std::size_t task_index;
        std::size_t node_index;
        bool counts_as_search;
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
    std::vector<std::unique_ptr<DirectInferencePipeline>> m_workers;
    std::vector<std::deque<PendingBatch>> m_pending;
    std::size_t m_nextWorker = 0;
    std::size_t m_nextTask = 0;
    std::mt19937 m_randomEngine;

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
        DirectInferencePipeline &worker = *m_workers[workerIndex];
        const DirectInferencePipeline::WritableBatch writable = worker.acquireWritableBatch();
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
                std::optional<std::size_t> leaf;
                if (!tree.root().expanded()) {
                    if (Game::isTerminal(tree.root().position)) {
                        tree.backPropagate(tree.rootIndex(),
                                           Game::terminalValue(tree.root().position).value_or(0.0F));
                        continue;
                    }
                    leaf = tree.rootIndex();
                    countsAsSearch = false;
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
                constexpr std::size_t encodedSize =
                    static_cast<std::size_t>(Game::inferenceDimensions().channels) *
                    static_cast<std::size_t>(Game::inferenceDimensions().rows) *
                    static_cast<std::size_t>(Game::inferenceDimensions().columns);
                Game::encodeInputInto(tree.node(*leaf).position,
                                      writable.data + leaves.size() * encodedSize);
                ++task.in_flight;
                leaves.push_back({*taskIndex, *leaf, countsAsSearch});
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
        DirectInferencePipeline &worker = *m_workers[workerIndex];
        bool outputReady = false;
        std::size_t processed = 0;
        try {
            const DirectInferenceOutput output = worker.waitCompleted(pending.slot_index);
            outputReady = true;
            const float *policies = output.policies.data_ptr<float>();
            const float *outcomes = output.outcomes.data_ptr<float>();
            for (; processed < pending.leaves.size(); ++processed) {
                const PendingLeaf &pendingLeaf = pending.leaves[processed];
                RootTask &task = tasks[pendingLeaf.task_index];
                Tree &tree = task.root.tree();
                const SearchInferenceResult<Game> inference = processSearchInference<Game>(
                    policies + processed * static_cast<std::size_t>(m_dimensions.actions),
                    outcomes + processed * static_cast<std::size_t>(m_dimensions.outcomes),
                    tree.node(pendingLeaf.node_index).position);
                tree.expand(pendingLeaf.node_index, inference);
                if (pendingLeaf.counts_as_search) {
                    tree.completeReservation(pendingLeaf.node_index, inference.value());
                } else {
                    tree.node(pendingLeaf.node_index).evaluating = false;
                    if (task.noise_pending) {
                        addNoise(task.root);
                        task.noise_pending = false;
                    }
                }
                --task.in_flight;
            }
            worker.release(pending.slot_index);
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
                if (pendingLeaf.counts_as_search) {
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
};
