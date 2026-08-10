#pragma once

#include "games/GameConcepts.hpp"
#include "search/InferenceTypes.hpp"
#include "util/py.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

// Owns the reusable game-generic MCTS arena, selection, expansion, backup, and rerooting.

template <typename Action> struct GameSearchEdge {
    Action action;
    float raw_prior;
    float prior;
    std::uint32_t visits = 0;
    float value_sum = 0.0F;
    float virtual_loss = 0.0F;
    std::optional<std::size_t> child_index;
};
template <SearchGame Game> struct GameSearchNode {
    using Position = typename Game::State;
    using Action = typename Game::Action;

    Position position;
    std::vector<GameSearchEdge<Action>> children;
    std::optional<std::size_t> parent_index;
    std::optional<std::size_t> parent_edge_index;
    std::uint32_t visits = 0;
    float value_sum = 0.0F;
    float virtual_loss = 0.0F;
    bool inference_pending = false;
    std::optional<WdlPrediction> network_outcome;

    [[nodiscard]] bool expanded() const noexcept { return !children.empty(); }
};

template <SearchGame Game> class GameSearchTree {
public:
    using Position = typename Game::State;
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
        for (const auto offset : range(initialCapacity)) {
            m_freeSlots.push_back(initialCapacity - offset - 1);
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
    selectAvailableLeaf(const float explorationConstant, const float fpuReduction = 0.0F,
                        const std::uint32_t minimumRootVisits = 0) {
        const std::size_t edgeCount = root().children.size();
        for (const auto edgeIndex : range(edgeCount)) {
            if (root().children[edgeIndex].visits >= minimumRootVisits) {
                continue;
            }
            const std::optional<std::size_t> leaf = selectAvailableLeaf(
                materializeChild(m_rootIndex, edgeIndex), explorationConstant, fpuReduction);
            if (leaf.has_value()) {
                return leaf;
            }
        }
        return selectAvailableLeaf(m_rootIndex, explorationConstant, fpuReduction);
    }

    void expand(const std::size_t nodeIndex, const SearchInferenceResult<Game> &inferenceResult) {
        Node &selected = node(nodeIndex);
        if (selected.expanded() || Game::isTerminal(selected.position)) {
            return;
        }
        selected.children.reserve(inferenceResult.actions.size());
        selected.network_outcome = inferenceResult.outcome;
        for (const auto &[action, prior] : inferenceResult.actions) {
            selected.children.push_back({
                .action = action,
                .raw_prior = prior,
                .prior = prior,
            });
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
        if (node(nodeIndex).inference_pending) {
            throw std::logic_error("Game search leaf is already reserved");
        }
        node(nodeIndex).inference_pending = true;
        updatePath(nodeIndex, 1, 0.0F, 1.0F);
    }

    void cancelReservation(const std::size_t nodeIndex) {
        if (!node(nodeIndex).inference_pending) {
            throw std::logic_error("Game search leaf is not reserved");
        }
        updatePath(nodeIndex, -1, 0.0F, -1.0F);
        node(nodeIndex).inference_pending = false;
    }

    void completeReservation(const std::size_t nodeIndex, const float value) {
        if (!node(nodeIndex).inference_pending) {
            throw std::logic_error("Game search leaf is not reserved");
        }
        completeReservedPath(nodeIndex, value);
        node(nodeIndex).inference_pending = false;
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
        for (const auto index : range(rootNode.children.size())) {
            rootNode.children[index].prior =
                std::lerp(rootNode.children[index].prior, noise[index] / sum, epsilon);
        }
    }

    [[nodiscard]] std::size_t actionIdToEdgeIndex(const int actionId) const {
        const Node &selected = root();
        for (const auto index : range(selected.children.size())) {
            if (Game::Encoding::actionId(selected.children[index].action, selected.position) ==
                actionId) {
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
        for (const auto index : range(edgeCount)) {
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
        for (const auto offset : range(m_nodes.size())) {
            const std::size_t slot = m_nodes.size() - offset - 1;
            m_nodes[slot].reset();
            m_freeSlots.push_back(slot);
        }
        m_liveNodeCount = 0;
        m_rootIndex = allocateNode(m_initialPosition, std::nullopt, std::nullopt);
    }

    void prepareForSearch(const std::uint32_t visitLimit, const std::uint32_t parallelSearches) {
        if (parallelSearches == 0) {
            throw std::invalid_argument("Parallel search count must be positive");
        }
        std::uint64_t maximumNewNodes = parallelSearches;
        if (root().visits < visitLimit) {
            maximumNewNodes =
                static_cast<std::uint64_t>(visitLimit - root().visits) + parallelSearches - 1U;
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

private:
    std::size_t m_maximumCapacity;
    Position m_initialPosition;
    std::vector<std::optional<Node>> m_nodes;
    std::vector<std::size_t> m_freeSlots;
    std::size_t m_liveNodeCount = 0;
    std::size_t m_rootIndex;
    float m_turnDiscount;

    [[nodiscard]] std::size_t bestEdgeIndex(const std::size_t nodeIndex,
                                            const float explorationConstant,
                                            const float fpuReduction) const {
        const Node &parent = node(nodeIndex);
        const float parentScale = std::sqrt(static_cast<float>(std::max(1U, parent.visits)));
        float bestScore = -std::numeric_limits<float>::infinity();
        std::size_t bestIndex = 0;
        for (const auto index : range(parent.children.size())) {
            const Edge &edge = parent.children[index];
            const float meanValue = childMeanValue(parent, edge, fpuReduction);
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

    [[nodiscard]] std::optional<std::size_t> selectAvailableLeaf(const std::size_t nodeIndex,
                                                                 const float explorationConstant,
                                                                 const float fpuReduction) {
        if (!node(nodeIndex).expanded()) {
            return node(nodeIndex).inference_pending ? std::nullopt
                                                     : std::optional<std::size_t>(nodeIndex);
        }
        const std::size_t edgeCount = node(nodeIndex).children.size();
        std::vector<bool> attempted(edgeCount, false);
        for (const auto attempt : range(edgeCount)) {
            static_cast<void>(attempt);
            float bestScore = -std::numeric_limits<float>::infinity();
            std::size_t bestIndex = 0;
            const float parentScale =
                std::sqrt(static_cast<float>(std::max(1U, node(nodeIndex).visits)));
            for (const auto index : range(edgeCount)) {
                if (attempted[index]) {
                    continue;
                }
                const Edge &edge = node(nodeIndex).children[index];
                const float meanValue = childMeanValue(node(nodeIndex), edge, fpuReduction);
                const float score = -meanValue + explorationConstant * edge.prior * parentScale /
                                                     (1.0F + edge.visits);
                if (score > bestScore) {
                    bestScore = score;
                    bestIndex = index;
                }
            }
            attempted[bestIndex] = true;
            const std::optional<std::size_t> leaf = selectAvailableLeaf(
                materializeChild(nodeIndex, bestIndex), explorationConstant, fpuReduction);
            if (leaf.has_value()) {
                return leaf;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] static float childMeanValue(const Node &parent, const Edge &edge,
                                              const float fpuReduction) {
        if (edge.visits > 0) {
            return (edge.value_sum + edge.virtual_loss) / static_cast<float>(edge.visits);
        }
        const float parentMean =
            parent.visits == 0 ? 0.0F : parent.value_sum / static_cast<float>(parent.visits);
        return -parentMean + fpuReduction;
    }

    void updatePath(std::size_t nodeIndex, const int visitDelta, float value,
                    const float virtualLossDelta) {
        while (true) {
            Node &selected = node(nodeIndex);
            selected.visits =
                static_cast<std::uint32_t>(static_cast<std::int64_t>(selected.visits) + visitDelta);
            selected.value_sum += value;
            selected.virtual_loss += virtualLossDelta;
            if (!selected.parent_index.has_value()) {
                break;
            }
            Edge &incoming = node(*selected.parent_index).children.at(*selected.parent_edge_index);
            incoming.visits =
                static_cast<std::uint32_t>(static_cast<std::int64_t>(incoming.visits) + visitDelta);
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
            Edge &incoming = node(*selected.parent_index).children.at(*selected.parent_edge_index);
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
        const Position child = Game::childState(node(parentIndex).position, edge.action);
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
            for (const auto offset : range(newCapacity - oldCapacity)) {
                m_freeSlots.push_back(newCapacity - offset - 1);
            }
        }
        if (m_freeSlots.empty()) {
            throw std::overflow_error("Game search tree capacity exhausted");
        }
        const std::size_t slot = m_freeSlots.back();
        m_freeSlots.pop_back();
        m_nodes[slot].emplace(Node{
            .position = std::move(position),
            .children = {},
            .parent_index = parentIndex,
            .parent_edge_index = parentEdgeIndex,
        });
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
        valueSum = std::clamp(valueSum, -static_cast<float>(visits), static_cast<float>(visits));
    }
};

template <SearchGame Game> class GameSearchRoot {
public:
    using Position = typename Game::State;
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
