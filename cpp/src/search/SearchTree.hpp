#pragma once

#include "games/GameConcepts.hpp"
#include "search/ForcedPlayouts.hpp"
#include "search/InferenceTypes.hpp"
#include "search/tree/RetainedStatistics.hpp"
#include "search/tree/TreeArena.hpp"
#include "search/tree/TreeSearchParameters.hpp"
#include "search/tree/TreeTypes.hpp"
#include "util/py.hpp"

#include <algorithm>
#include <array>
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

// Provides the game-generic MCTS selection, expansion, backup, and retained-root facade.

template <SearchGame Game> class GameSearchTree {
public:
    using Position = typename Game::State;
    using Action = typename Game::Action;
    using Node = GameSearchNode<Game>;
    using Edge = GameSearchEdge<Action>;

    // Chess tops out at 218 legal moves and Go 9x9 at 82; the selection bitmask is sized from this.
    static constexpr std::size_t maximumBranchingFactor = 256;

    GameSearchTree(Position rootPosition, const std::size_t initialCapacity,
                   const std::size_t maximumCapacity = 0, const float valueDiscountPerPly = 1.0F)
        : m_arena(std::move(rootPosition), initialCapacity, maximumCapacity),
          m_valueDiscountPerPly(valueDiscountPerPly) {
        if (!std::isfinite(valueDiscountPerPly) || valueDiscountPerPly <= 0.0F ||
            valueDiscountPerPly > 1.0F) {
            throw std::invalid_argument("Game search value discount per ply must be in (0, 1]");
        }
    }

    [[nodiscard]] const Node &node(const std::size_t index) const { return m_arena.node(index); }
    [[nodiscard]] const Position &position(const std::size_t index) const {
        return m_arena.position(index);
    }
    [[nodiscard]] const Position &rootPosition() const { return m_arena.rootPosition(); }
    [[nodiscard]] Node &node(const std::size_t index) { return m_arena.node(index); }
    [[nodiscard]] const Node &root() const { return m_arena.root(); }
    [[nodiscard]] Node &root() { return m_arena.root(); }
    [[nodiscard]] std::size_t rootIndex() const noexcept { return m_arena.rootIndex(); }
    [[nodiscard]] std::size_t liveNodeCount() const noexcept { return m_arena.liveNodeCount(); }
    [[nodiscard]] std::size_t capacity() const noexcept { return m_arena.capacity(); }
    [[nodiscard]] std::size_t totalChildCount() const noexcept { return m_arena.totalChildCount(); }
    [[nodiscard]] float valueDiscountPerPly() const noexcept { return m_valueDiscountPerPly; }

    [[nodiscard]] std::optional<std::size_t>
    selectAvailableLeaf(const TreeSearchParameters &parameters,
                        const bool forceRootPlayouts = false) {
        const std::size_t edgeCount = root().children.size();
        for (const auto edgeIndex : range(edgeCount)) {
            const Edge &edge = root().children[edgeIndex];
            if (!requiresForcedRootVisit(
                    {.prior = edge.prior,
                     .visits = edge.visits,
                     .child_mean_value = -edgeValueForParent(root(), edge, parameters)},
                    root().visits,
                    forceRootPlayouts ? parameters.forced_playout_coefficient : 0.0F)) {
                continue;
            }
            const std::optional<std::size_t> leaf =
                selectAvailableLeaf(m_arena.materializeChild(rootIndex(), edgeIndex), parameters);
            if (leaf.has_value()) {
                return leaf;
            }
        }
        return selectAvailableLeaf(rootIndex(), parameters);
    }

    [[nodiscard]] std::vector<std::uint32_t>
    policyTargetVisits(const float explorationConstant, const bool forcedPlayoutsEnabled) const {
        const Node &rootNode = root();
        std::vector<RootChildSearchStatistics> children;
        children.reserve(rootNode.children.size());
        for (const Edge &edge : rootNode.children) {
            children.push_back({
                .prior = edge.prior,
                .visits = edge.visits,
                .child_mean_value = edge.visits == 0 ? 0.0F
                                                     : m_valueDiscountPerPly * edge.value_sum /
                                                           static_cast<float>(edge.visits),
            });
        }
        return prunedRootPolicyVisits(children, rootNode.visits, explorationConstant,
                                      forcedPlayoutsEnabled);
    }

    void expand(const std::size_t nodeIndex, const SearchInferenceResult<Game> &inferenceResult) {
        Node &selected = node(nodeIndex);
        if (selected.expanded() || Game::isTerminal(m_arena.position(nodeIndex))) {
            return;
        }
        selected.children.reserve(inferenceResult.actions.size());
        selected.network_outcome = inferenceResult.outcome;
        selected.search_correction = inferenceResult.search_correction;
        for (const ScoredAction<Action> &scored : inferenceResult.actions) {
            selected.children.push_back({
                .action = scored.action,
                .action_id = scored.action_id,
                .raw_prior = scored.prior,
                .prior = scored.prior,
            });
        }
    }

    void backPropagate(std::size_t nodeIndex, float value) {
        while (true) {
            Node &selected = m_arena.liveNode(nodeIndex);
            ++selected.visits;
            selected.value_sum += value;
            if (!selected.parent_index.has_value()) {
                break;
            }
            Node &parent = m_arena.liveNode(*selected.parent_index);
            Edge &incoming = parent.children[*selected.parent_edge_index];
            ++incoming.visits;
            incoming.value_sum += value;
            value = -value * m_valueDiscountPerPly;
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
                std::lerp(rootNode.children[index].raw_prior, noise[index] / sum, epsilon);
        }
    }

    [[nodiscard]] std::size_t actionIdToEdgeIndex(const int actionId) const {
        const Node &selected = root();
        for (const auto index : range(selected.children.size())) {
            if (selected.children[index].action_id == actionId) {
                return index;
            }
        }
        throw std::invalid_argument("Selected action is not a root child");
    }

    void reroot(const int actionId) { rerootEdge(actionIdToEdgeIndex(actionId)); }
    void rerootEdge(const std::size_t edgeIndex) { m_arena.rerootEdge(edgeIndex); }
    void reset() { m_arena.reset(); }

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
        m_arena.pruneToLiveNodeLimit(
            std::max<std::size_t>(1, capacity() - static_cast<std::size_t>(maximumNewNodes) - 1));
    }

    void discount(const float retainedFraction) {
        if (retainedFraction < 0.0F || retainedFraction > 1.0F) {
            throw std::invalid_argument("Game search discount must be in [0, 1]");
        }
        m_arena.requireIdle();
        const std::uint32_t retainedRootVisits = static_cast<std::uint32_t>(
            static_cast<double>(root().visits) * static_cast<double>(retainedFraction));
        search_tree_detail::retainSubtree(m_arena, rootIndex(), retainedRootVisits,
                                          m_valueDiscountPerPly);
        const std::size_t liveNodeLimit = static_cast<std::size_t>(retainedRootVisits) + 1;
        m_arena.pruneToLiveNodeLimit(std::max<std::size_t>(1, liveNodeLimit));
    }

    [[nodiscard]] int maximumDepth() const {
        int result = 1;
        std::vector<std::pair<std::size_t, int>> pending = {{rootIndex(), 1}};
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
    search_tree_detail::TreeArena<Game> m_arena;
    float m_valueDiscountPerPly;

    [[nodiscard]] std::optional<std::size_t>
    selectAvailableLeaf(const std::size_t nodeIndex, const TreeSearchParameters &parameters) {
        const Node &selected = m_arena.liveNode(nodeIndex);
        if (!selected.expanded()) {
            return selected.inference_pending ? std::nullopt
                                              : std::optional<std::size_t>(nodeIndex);
        }
        const std::size_t edgeCount = selected.children.size();
        if (edgeCount > maximumBranchingFactor) {
            throw std::logic_error("Game search node exceeds the supported branching factor");
        }
        // A stack bitmask: the descent allocated one std::vector<bool> per node per simulation.
        std::array<std::uint64_t, maximumBranchingFactor / 64> attempted{};
        for (const auto attempt : range(edgeCount)) {
            static_cast<void>(attempt);
            const Node &parent = m_arena.liveNode(nodeIndex);
            const float parentScale = std::sqrt(static_cast<float>(std::max(1U, parent.visits)));
            float bestScore = -std::numeric_limits<float>::infinity();
            std::size_t bestIndex = 0;
            for (const auto index : range(edgeCount)) {
                if ((attempted[index / 64] >> (index % 64) & 1U) != 0U) {
                    continue;
                }
                const Edge &edge = parent.children[index];
                const float score = edgeValueForParent(parent, edge, parameters) +
                                    parameters.exploration_constant * edge.prior * parentScale /
                                        (1.0F + edge.visits);
                if (score > bestScore) {
                    bestScore = score;
                    bestIndex = index;
                }
            }
            attempted[bestIndex / 64] |= std::uint64_t{1} << (bestIndex % 64);
            const std::optional<std::size_t> leaf =
                selectAvailableLeaf(m_arena.materializeChild(nodeIndex, bestIndex), parameters);
            if (leaf.has_value()) {
                return leaf;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] float edgeValueForParent(const Node &parent, const Edge &edge,
                                           const TreeSearchParameters &parameters) const {
        if (edge.visits > 0) {
            return (-m_valueDiscountPerPly * edge.value_sum -
                    parameters.virtual_loss_weight * edge.virtual_loss) /
                   static_cast<float>(edge.visits);
        }
        const float parentMean =
            parent.visits == 0 ? 0.0F : parent.value_sum / static_cast<float>(parent.visits);
        return parameters.first_play_urgency.unvisitedParentValue(parentMean);
    }

    void updatePath(std::size_t nodeIndex, const int visitDelta, float value,
                    const float virtualLossDelta) {
        while (true) {
            Node &selected = m_arena.liveNode(nodeIndex);
            selected.visits =
                static_cast<std::uint32_t>(static_cast<std::int64_t>(selected.visits) + visitDelta);
            selected.value_sum += value;
            selected.virtual_loss += virtualLossDelta;
            if (!selected.parent_index.has_value()) {
                break;
            }
            Edge &incoming =
                m_arena.liveNode(*selected.parent_index).children[*selected.parent_edge_index];
            incoming.visits =
                static_cast<std::uint32_t>(static_cast<std::int64_t>(incoming.visits) + visitDelta);
            incoming.value_sum += value;
            incoming.virtual_loss += virtualLossDelta;
            value = -value * m_valueDiscountPerPly;
            nodeIndex = *selected.parent_index;
        }
    }

    void completeReservedPath(std::size_t nodeIndex, float value) {
        while (true) {
            Node &selected = m_arena.liveNode(nodeIndex);
            selected.value_sum += value;
            selected.virtual_loss -= 1.0F;
            if (!selected.parent_index.has_value()) {
                break;
            }
            Edge &incoming =
                m_arena.liveNode(*selected.parent_index).children[*selected.parent_edge_index];
            incoming.value_sum += value;
            incoming.virtual_loss -= 1.0F;
            value = -value * m_valueDiscountPerPly;
            nodeIndex = *selected.parent_index;
        }
    }
};

template <SearchGame Game> class GameSearchRoot {
public:
    using Position = typename Game::State;
    using Tree = GameSearchTree<Game>;

    GameSearchRoot(Position position, const std::size_t initialCapacity,
                   const std::size_t maximumCapacity = 0, const float valueDiscountPerPly = 1.0F)
        : m_tree(std::make_shared<Tree>(std::move(position), initialCapacity, maximumCapacity,
                                        valueDiscountPerPly)) {}

    [[nodiscard]] const Position &position() const { return m_tree->rootPosition(); }
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
