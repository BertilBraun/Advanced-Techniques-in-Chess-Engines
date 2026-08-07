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
#include <numeric>
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
    std::optional<WdlPrediction> network_outcome;

    [[nodiscard]] bool expanded() const noexcept { return !children.empty(); }
};

template <typename Game> class GameSearchTree {
public:
    using Position = typename Game::Position;
    using Action = typename Game::Action;
    using Node = GameSearchNode<Game>;
    using Edge = GameSearchEdge<Action>;

    GameSearchTree(Position rootPosition, const std::size_t initialCapacity,
                   const std::size_t maximumCapacity = 0, const float turnDiscount = 1.0F)
        : m_maximumCapacity(maximumCapacity == 0 ? initialCapacity : maximumCapacity),
          m_initialPosition(rootPosition), m_turnDiscount(turnDiscount) {
        if (initialCapacity == 0 || m_maximumCapacity < initialCapacity) {
            throw std::invalid_argument("Game search tree capacity must be positive");
        }
        if (turnDiscount <= 0.0F || turnDiscount > 1.0F) {
            throw std::invalid_argument("Game search turn discount must be in (0, 1]");
        }
        m_nodes.reserve(initialCapacity);
        m_nodes.resize(initialCapacity);
        for (std::size_t slot = initialCapacity; slot > 0; --slot) {
            m_freeSlots.push_back(slot - 1);
        }
        m_rootIndex = allocateNode(std::move(rootPosition), std::nullopt, std::nullopt);
    }

    [[nodiscard]] const Node &node(const std::size_t index) const {
        return const_cast<GameSearchTree *>(this)->node(index);
    }
    [[nodiscard]] Node &node(const std::size_t index) {
        if (index >= m_nodes.size() || !m_nodes[index].has_value()) {
            throw std::out_of_range("Game search node is not live");
        }
        return *m_nodes[index];
    }
    [[nodiscard]] const Node &root() const { return node(m_rootIndex); }
    [[nodiscard]] Node &root() { return node(m_rootIndex); }
    [[nodiscard]] std::size_t rootIndex() const noexcept { return m_rootIndex; }
    [[nodiscard]] std::size_t liveNodeCount() const noexcept { return m_liveNodeCount; }
    [[nodiscard]] std::size_t capacity() const noexcept { return m_nodes.size(); }
    [[nodiscard]] std::size_t maximumCapacity() const noexcept { return m_maximumCapacity; }
    [[nodiscard]] std::size_t totalChildCount() const noexcept {
        std::size_t count = 0;
        for (const std::optional<Node> &slot : m_nodes) {
            if (slot.has_value()) {
                count += slot->children.size();
            }
        }
        return count;
    }

    [[nodiscard]] std::optional<std::size_t>
    selectAvailableLeaf(const float explorationConstant, const std::uint32_t minimumRootVisits = 0) {
        const std::size_t edgeCount = root().children.size();
        for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex) {
            if (root().children[edgeIndex].visits >= minimumRootVisits) {
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
        selected.network_outcome = inferenceResult.outcome;
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
            value = -value * m_turnDiscount;
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
        rerootEdge(edgeIndex);
    }

    void rerootEdge(const std::size_t edgeIndex) {
        if (edgeIndex >= root().children.size()) {
            throw std::out_of_range("Cannot reroot to a missing game-search edge");
        }
        const std::size_t oldRootIndex = m_rootIndex;
        const std::size_t retainedIndex = materializeChild(oldRootIndex, edgeIndex);
        const std::size_t edgeCount = node(oldRootIndex).children.size();
        for (std::size_t index = 0; index < edgeCount; ++index) {
            Edge &discarded = node(oldRootIndex).children[index];
            if (index != edgeIndex && discarded.child_index.has_value()) {
                reclaimSubtree(*discarded.child_index);
                discarded.child_index.reset();
            }
        }
        node(oldRootIndex).children[edgeIndex].child_index.reset();
        Node &retainedRoot = node(retainedIndex);
        retainedRoot.parent_index.reset();
        retainedRoot.parent_edge_index.reset();
        m_rootIndex = retainedIndex;
        m_initialPosition = retainedRoot.position;
        releaseNode(oldRootIndex);
    }

    void reset() {
        m_freeSlots.clear();
        for (std::size_t slot = m_nodes.size(); slot > 0; --slot) {
            m_nodes[slot - 1].reset();
            m_freeSlots.push_back(slot - 1);
        }
        m_liveNodeCount = 0;
        m_rootIndex = allocateNode(m_initialPosition, std::nullopt, std::nullopt);
    }

    void prepareForSearch(const std::uint32_t visitLimit,
                          const std::uint32_t parallelSearches) {
        if (parallelSearches == 0) {
            throw std::invalid_argument("Parallel search count must be positive");
        }
        std::uint64_t maximumNewNodes = parallelSearches;
        if (root().visits < visitLimit) {
            maximumNewNodes = static_cast<std::uint64_t>(visitLimit - root().visits) +
                              parallelSearches - 1U;
        }
        if (maximumNewNodes + 1U >= capacity()) {
            throw std::logic_error("Game search arena cannot reserve search and reroot slots");
        }
        pruneToLiveNodeLimit(
            std::max<std::size_t>(1, capacity() - static_cast<std::size_t>(maximumNewNodes) - 1));
    }

    void discount(const float retainedFraction) {
        if (retainedFraction < 0.0F || retainedFraction > 1.0F) {
            throw std::invalid_argument("Game search discount must be in [0, 1]");
        }
        for (std::optional<Node> &slot : m_nodes) {
            if (!slot.has_value()) {
                continue;
            }
            discountStatistics(slot->visits, slot->value_sum, retainedFraction);
            for (Edge &edge : slot->children) {
                discountStatistics(edge.visits, edge.value_sum, retainedFraction);
            }
        }
        const std::size_t liveNodeLimit = static_cast<std::size_t>(root().visits) + 1;
        pruneToLiveNodeLimit(std::max<std::size_t>(1, liveNodeLimit));
    }

    [[nodiscard]] int maximumDepth() const {
        int result = 1;
        std::vector<std::pair<std::size_t, int>> pending = {{m_rootIndex, 1}};
        while (!pending.empty()) {
            const auto [nodeIndex, depth] = pending.back();
            pending.pop_back();
            result = std::max(result, depth);
            for (const Edge &edge : node(nodeIndex).children) {
                if (edge.child_index.has_value()) {
                    pending.emplace_back(*edge.child_index, depth + 1);
                }
            }
        }
        return result;
    }

    [[nodiscard]] std::uint32_t evaluatingNodeCount() const {
        return static_cast<std::uint32_t>(
            std::ranges::count_if(m_nodes, [](const std::optional<Node> &slot) {
                return slot.has_value() && slot->evaluating;
            }));
    }

    [[nodiscard]] float totalVirtualLoss() const {
        float result = root().virtual_loss;
        for (const std::optional<Node> &slot : m_nodes) {
            if (!slot.has_value()) {
                continue;
            }
            for (const Edge &edge : slot->children) {
                result += edge.virtual_loss;
            }
        }
        return result;
    }

    [[nodiscard]] std::size_t preferredRootEdge() const {
        const Node &rootNode = root();
        if (rootNode.children.empty()) {
            throw std::logic_error("Cannot choose an action from an unexpanded root");
        }
        std::size_t preferred = 0;
        for (std::size_t index = 1; index < rootNode.children.size(); ++index) {
            const Edge &candidate = rootNode.children[index];
            const Edge &current = rootNode.children[preferred];
            if (candidate.visits > current.visits ||
                (candidate.visits == current.visits &&
                 (candidate.prior > current.prior ||
                  (candidate.prior == current.prior &&
                   Game::actionId(candidate.action, rootNode.position) <
                       Game::actionId(current.action, rootNode.position))))) {
                preferred = index;
            }
        }
        return preferred;
    }

private:
    std::size_t m_maximumCapacity;
    Position m_initialPosition;
    std::vector<std::optional<Node>> m_nodes;
    std::vector<std::size_t> m_freeSlots;
    std::size_t m_liveNodeCount = 0;
    std::size_t m_rootIndex;
    float m_turnDiscount;

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
        if (!node(nodeIndex).expanded()) {
            return node(nodeIndex).evaluating ? std::nullopt
                                       : std::optional<std::size_t>(nodeIndex);
        }
        const std::size_t edgeCount = node(nodeIndex).children.size();
        std::vector<bool> attempted(edgeCount, false);
        for (std::size_t attempt = 0; attempt < edgeCount; ++attempt) {
            float bestScore = -std::numeric_limits<float>::infinity();
            std::size_t bestIndex = 0;
            const float parentScale =
                std::sqrt(static_cast<float>(std::max(1U, node(nodeIndex).visits)));
            for (std::size_t index = 0; index < edgeCount; ++index) {
                if (attempted[index]) {
                    continue;
                }
                const Edge &edge = node(nodeIndex).children[index];
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
            value = -value * m_turnDiscount;
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
            value = -value * m_turnDiscount;
            nodeIndex = *selected.parent_index;
        }
    }

    [[nodiscard]] std::size_t materializeChild(const std::size_t parentIndex,
                                               const std::size_t edgeIndex) {
        Edge &edge = node(parentIndex).children.at(edgeIndex);
        if (edge.child_index.has_value()) {
            return *edge.child_index;
        }
        const Position child = Game::childPosition(node(parentIndex).position, edge.action);
        const std::size_t childIndex = allocateNode(child, parentIndex, edgeIndex);
        node(parentIndex).children[edgeIndex].child_index = childIndex;
        return childIndex;
    }

    [[nodiscard]] std::size_t allocateNode(Position position,
                                           const std::optional<std::size_t> parentIndex,
                                           const std::optional<std::size_t> parentEdgeIndex) {
        if (m_freeSlots.empty() && m_nodes.size() < m_maximumCapacity) {
            const std::size_t oldCapacity = m_nodes.size();
            const std::size_t newCapacity =
                std::min(m_maximumCapacity, std::max(oldCapacity + 1, oldCapacity * 2));
            m_nodes.resize(newCapacity);
            for (std::size_t slot = newCapacity; slot > oldCapacity; --slot) {
                m_freeSlots.push_back(slot - 1);
            }
        }
        if (m_freeSlots.empty()) {
            throw std::overflow_error("Game search tree capacity exhausted");
        }
        const std::size_t slot = m_freeSlots.back();
        m_freeSlots.pop_back();
        m_nodes[slot].emplace(
            Node{std::move(position), {}, parentIndex, parentEdgeIndex});
        ++m_liveNodeCount;
        return slot;
    }

    void releaseNode(const std::size_t index) {
        if (index == m_rootIndex) {
            throw std::logic_error("Cannot release the active game-search root");
        }
        static_cast<void>(node(index));
        m_nodes[index].reset();
        m_freeSlots.push_back(index);
        --m_liveNodeCount;
    }

    void reclaimSubtree(const std::size_t index) {
        std::vector<std::pair<std::size_t, bool>> pending = {{index, false}};
        while (!pending.empty()) {
            const auto [currentIndex, visited] = pending.back();
            pending.pop_back();
            if (visited) {
                releaseNode(currentIndex);
                continue;
            }
            pending.emplace_back(currentIndex, true);
            for (const Edge &edge : node(currentIndex).children) {
                if (edge.child_index.has_value()) {
                    pending.emplace_back(*edge.child_index, false);
                }
            }
        }
    }

    void pruneToLiveNodeLimit(const std::size_t liveNodeLimit) {
        if (m_liveNodeCount <= liveNodeLimit) {
            return;
        }
        std::vector<std::pair<std::size_t, bool>> pending = {{m_rootIndex, false}};
        std::vector<std::size_t> postOrder;
        while (!pending.empty()) {
            const auto [currentIndex, visited] = pending.back();
            pending.pop_back();
            if (visited) {
                if (currentIndex != m_rootIndex) {
                    postOrder.push_back(currentIndex);
                }
                continue;
            }
            pending.emplace_back(currentIndex, true);
            for (const Edge &edge : node(currentIndex).children) {
                if (edge.child_index.has_value()) {
                    pending.emplace_back(*edge.child_index, false);
                }
            }
        }
        for (const std::size_t index : postOrder) {
            if (m_liveNodeCount <= liveNodeLimit) {
                break;
            }
            Node &pruned = node(index);
            node(*pruned.parent_index).children[*pruned.parent_edge_index].child_index.reset();
            releaseNode(index);
        }
    }

    static void discountStatistics(std::uint32_t &visits, float &valueSum,
                                   const float retainedFraction) {
        visits = static_cast<std::uint32_t>(static_cast<float>(visits) * retainedFraction);
        valueSum *= retainedFraction;
        valueSum = std::clamp(valueSum, -static_cast<float>(visits),
                              static_cast<float>(visits));
    }
};

template <typename Game> class GameSearchRoot {
public:
    using Position = typename Game::Position;
    using Tree = GameSearchTree<Game>;

    GameSearchRoot(Position position, const std::size_t initialCapacity,
                   const std::size_t maximumCapacity = 0, const float turnDiscount = 1.0F)
        : m_tree(std::make_shared<Tree>(std::move(position), initialCapacity, maximumCapacity,
                                       turnDiscount)) {}

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

template <typename Game> struct GameSearchRequest {
    GameSearchRoot<Game> root;
    std::uint32_t visit_limit;
    bool add_root_noise;
    bool count_root_initialization = false;
};

struct GameSearchBatchResult {
    std::vector<GameSearchResult> results;
    std::uint64_t simulations_completed;
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
                      const std::uint64_t modelGeneration, const bool resetTreesOnRefresh = true,
                      const float turnDiscount = 1.0F)
        : m_inferenceParameters(inferenceParameters), m_searchParameters(searchParameters),
          m_dimensions(Game::inferenceDimensions()), m_modelGeneration(modelGeneration),
          m_pending(inferenceParameters.workers), m_randomEngine(std::random_device{}()),
          m_batchHistogram(inferenceParameters.batch_size + 1, 0),
          m_resetTreesOnRefresh(resetTreesOnRefresh), m_turnDiscount(turnDiscount) {
        m_workers.reserve(inferenceParameters.workers);
        for (std::size_t worker = 0; worker < inferenceParameters.workers; ++worker) {
            m_workers.push_back(std::make_unique<DirectInferencePipeline>(
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
        DirectInferencePipeline &worker = *m_workers.front();
        constexpr std::size_t encodedSize =
            static_cast<std::size_t>(Game::inferenceDimensions().channels) *
            static_cast<std::size_t>(Game::inferenceDimensions().rows) *
            static_cast<std::size_t>(Game::inferenceDimensions().columns);
        while (offset < positions.size()) {
            const std::size_t batchSize =
                std::min(m_inferenceParameters.batch_size, positions.size() - offset);
            const DirectInferencePipeline::WritableBatch writable = worker.acquireWritableBatch();
            for (std::size_t row = 0; row < batchSize; ++row) {
                Game::encodeInputInto(positions[offset + row],
                                      writable.data + row * encodedSize);
            }
            worker.submit(writable.slotIndex, batchSize);
            bool outputReady = false;
            try {
                const DirectInferenceOutput output = worker.waitCompleted(writable.slotIndex);
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
        for (const std::unique_ptr<DirectInferencePipeline> &worker : m_workers) {
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
        for (const std::unique_ptr<DirectInferencePipeline> &worker : m_workers) {
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
        for (const std::unique_ptr<DirectInferencePipeline> &worker : m_workers) {
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
    std::vector<std::unique_ptr<DirectInferencePipeline>> m_workers;
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
        DirectInferencePipeline &worker = *m_workers[workerIndex];
        bool outputReady = false;
        std::size_t processed = 0;
        try {
            const auto waitStartedAt = std::chrono::steady_clock::now();
            const DirectInferenceOutput output = worker.waitCompleted(pending.slot_index);
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
