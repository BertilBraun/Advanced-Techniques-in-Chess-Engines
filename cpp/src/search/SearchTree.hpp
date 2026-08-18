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
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

// Provides the game-generic MCTS selection, expansion, backup, and retained-root facade.

template <SearchGame Game> class GameSearchTree {
public:
    using Position = typename Game::State;
    using Action = typename Game::Action;
    using Node = GameSearchNode<Game>;
    using Edge = GameSearchEdge<Action>;

    GameSearchTree(Position rootPosition, const std::size_t initialCapacity,
                   const std::size_t maximumCapacity = 0, const float valueDiscountPerPly = 1.0F,
                   SearchAlgorithmParameters algorithm = MonteCarloTreeSearchParameters{})
        : m_arena(std::move(rootPosition), initialCapacity, maximumCapacity, std::move(algorithm)),
          m_valueDiscountPerPly(valueDiscountPerPly) {
        if (!std::isfinite(valueDiscountPerPly) || valueDiscountPerPly <= 0.0F ||
            valueDiscountPerPly > 1.0F) {
            throw std::invalid_argument("Game search value discount per ply must be in (0, 1]");
        }
    }

    [[nodiscard]] const Node &node(const std::size_t index) const { return m_arena.node(index); }
    [[nodiscard]] Node &node(const std::size_t index) { return m_arena.node(index); }
    [[nodiscard]] const Node &root() const { return m_arena.root(); }
    [[nodiscard]] Node &root() { return m_arena.root(); }
    [[nodiscard]] std::size_t rootIndex() const noexcept { return m_arena.rootIndex(); }
    [[nodiscard]] std::size_t liveNodeCount() const noexcept { return m_arena.liveNodeCount(); }
    [[nodiscard]] std::size_t capacity() const noexcept { return m_arena.capacity(); }
    [[nodiscard]] std::size_t totalChildCount() const noexcept { return m_arena.totalChildCount(); }
    [[nodiscard]] float valueDiscountPerPly() const noexcept { return m_valueDiscountPerPly; }
    [[nodiscard]] bool graphSearch() const noexcept { return m_arena.graphSearch(); }
    [[nodiscard]] const GraphSearchStatistics &graphStatistics() const noexcept {
        return m_arena.graphStatistics();
    }
    [[nodiscard]] GraphStructureStatistics graphStructureStatistics() const {
        return m_arena.graphStructureStatistics();
    }

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

    [[nodiscard]] std::optional<GraphSearchSelection>
    selectAvailableGraphLeaf(const TreeSearchParameters &parameters,
                             const bool forceRootPlayouts = false) {
        if (!graphSearch()) {
            throw std::logic_error("Graph selection requires a graph-search root");
        }
        std::vector<std::size_t> trajectoryNodes = {rootIndex()};
        trajectoryNodes.reserve(128);
        GameSearchPath path{.steps = {}, .leaf_index = rootIndex()};
        path.steps.reserve(128);
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
            if (const std::optional<GraphSearchSelection> selected =
                    selectGraphEdge(rootIndex(), edgeIndex, parameters, path, trajectoryNodes);
                selected.has_value()) {
                return selected;
            }
        }
        return selectAvailableGraphLeaf(rootIndex(), parameters, path, trajectoryNodes);
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
        m_arena.recordEdgesCreated(inferenceResult.actions.size());
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
            value = -value * m_valueDiscountPerPly;
            nodeIndex = *selected.parent_index;
        }
    }

    void backPropagateGraph(const GameSearchPath &path, float value, const bool updateLeaf) {
        if (!graphSearch()) {
            throw std::logic_error("Graph backup requires a graph-search root");
        }
        if (updateLeaf) {
            Node &leaf = node(path.leaf_index);
            ++leaf.visits;
            leaf.value_sum += value;
        }
        bool finalUnlinkedValue = !updateLeaf;
        for (const GameSearchPathStep &step : std::ranges::reverse_view(path.steps)) {
            Node &parent = node(step.node_index);
            Edge &edge = parent.children.at(step.edge_index);
            if (!finalUnlinkedValue && edge.child_index.has_value() &&
                node(*edge.child_index).incoming_edges > 1) {
                value = graphCorrection(edge, node(*edge.child_index));
            }
            finalUnlinkedValue = false;
            ++edge.visits;
            edge.value_sum += value;
            value = -value * m_valueDiscountPerPly;
            ++parent.visits;
            parent.value_sum += value;
        }
    }

    void reserve(const std::size_t nodeIndex) {
        if (node(nodeIndex).inference_pending) {
            throw std::logic_error("Game search leaf is already reserved");
        }
        node(nodeIndex).inference_pending = true;
        updatePath(nodeIndex, 1, 0.0F, 1.0F);
    }

    void reserveGraph(const GameSearchPath &path) {
        Node &leaf = node(path.leaf_index);
        if (leaf.inference_pending) {
            throw std::logic_error("Game search graph leaf is already reserved");
        }
        leaf.inference_pending = true;
        ++leaf.visits;
        leaf.virtual_loss += 1.0F;
        for (const GameSearchPathStep &step : std::ranges::reverse_view(path.steps)) {
            Edge &edge = node(step.node_index).children.at(step.edge_index);
            ++edge.visits;
            edge.virtual_loss += 1.0F;
            Node &parent = node(step.node_index);
            ++parent.visits;
            parent.virtual_loss += 1.0F;
        }
    }

    void cancelReservation(const std::size_t nodeIndex) {
        if (!node(nodeIndex).inference_pending) {
            throw std::logic_error("Game search leaf is not reserved");
        }
        updatePath(nodeIndex, -1, 0.0F, -1.0F);
        node(nodeIndex).inference_pending = false;
    }

    void cancelGraphReservation(const GameSearchPath &path) {
        Node &leaf = node(path.leaf_index);
        if (!leaf.inference_pending) {
            throw std::logic_error("Game search graph leaf is not reserved");
        }
        --leaf.visits;
        leaf.virtual_loss -= 1.0F;
        for (const GameSearchPathStep &step : std::ranges::reverse_view(path.steps)) {
            Edge &edge = node(step.node_index).children.at(step.edge_index);
            --edge.visits;
            edge.virtual_loss -= 1.0F;
            Node &parent = node(step.node_index);
            --parent.visits;
            parent.virtual_loss -= 1.0F;
        }
        leaf.inference_pending = false;
    }

    void completeReservation(const std::size_t nodeIndex, const float value) {
        if (!node(nodeIndex).inference_pending) {
            throw std::logic_error("Game search leaf is not reserved");
        }
        completeReservedPath(nodeIndex, value);
        node(nodeIndex).inference_pending = false;
    }

    void completeGraphReservation(const GameSearchPath &path, float value) {
        Node &leaf = node(path.leaf_index);
        if (!leaf.inference_pending) {
            throw std::logic_error("Game search graph leaf is not reserved");
        }
        leaf.value_sum += value;
        leaf.virtual_loss -= 1.0F;
        for (const GameSearchPathStep &step : std::ranges::reverse_view(path.steps)) {
            Edge &edge = node(step.node_index).children.at(step.edge_index);
            if (edge.child_index.has_value() && node(*edge.child_index).incoming_edges > 1) {
                value = graphCorrection(edge, node(*edge.child_index));
            }
            edge.value_sum += value;
            edge.virtual_loss -= 1.0F;
            value = -value * m_valueDiscountPerPly;
            Node &parent = node(step.node_index);
            parent.value_sum += value;
            parent.virtual_loss -= 1.0F;
        }
        leaf.inference_pending = false;
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
            if (Game::Encoding::actionId(selected.children[index].action, selected.position) ==
                actionId) {
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
        if (graphSearch()) {
            discountGraph(retainedFraction);
            return;
        }
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
        std::unordered_set<std::size_t> visited;
        while (!pending.empty()) {
            const auto [nodeIndex, depth] = pending.back();
            pending.pop_back();
            if (!visited.insert(nodeIndex).second) {
                continue;
            }
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
                const float score = edgeValueForParent(node(nodeIndex), edge, parameters) +
                                    parameters.exploration_constant * edge.prior * parentScale /
                                        (1.0F + edge.visits);
                if (score > bestScore) {
                    bestScore = score;
                    bestIndex = index;
                }
            }
            attempted[bestIndex] = true;
            const std::optional<std::size_t> leaf =
                selectAvailableLeaf(m_arena.materializeChild(nodeIndex, bestIndex), parameters);
            if (leaf.has_value()) {
                return leaf;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<GraphSearchSelection>
    selectAvailableGraphLeaf(const std::size_t nodeIndex, const TreeSearchParameters &parameters,
                             GameSearchPath &path, std::vector<std::size_t> &trajectoryNodes) {
        if (!node(nodeIndex).expanded()) {
            if (node(nodeIndex).inference_pending) {
                return std::nullopt;
            }
            path.leaf_index = nodeIndex;
            return GraphSearchSelection{
                .path = path, .immediate_value = std::nullopt, .update_leaf = true};
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
                const float score = edgeValueForParent(node(nodeIndex), edge, parameters) +
                                    parameters.exploration_constant * edge.prior * parentScale /
                                        (1.0F + edge.visits);
                if (score > bestScore) {
                    bestScore = score;
                    bestIndex = index;
                }
            }
            attempted[bestIndex] = true;
            if (const std::optional<GraphSearchSelection> selected =
                    selectGraphEdge(nodeIndex, bestIndex, parameters, path, trajectoryNodes);
                selected.has_value()) {
                return selected;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<GraphSearchSelection>
    selectGraphEdge(const std::size_t nodeIndex, const std::size_t edgeIndex,
                    const TreeSearchParameters &parameters, GameSearchPath &path,
                    std::vector<std::size_t> &trajectoryNodes) {
        const search_tree_detail::GraphChildMaterialization materialized =
            m_arena.materializeGraphChild(nodeIndex, edgeIndex, trajectoryNodes);
        path.steps.push_back({.node_index = nodeIndex, .edge_index = edgeIndex});
        path.leaf_index = materialized.node_index;
        if (materialized.cycle) {
            return GraphSearchSelection{
                .path = path,
                .immediate_value = Game::cycleValue(node(materialized.node_index).position),
                .update_leaf = false};
        }
        Node &child = node(materialized.node_index);
        Edge &edge = node(nodeIndex).children[edgeIndex];
        const auto &graphParameters =
            std::get<MonteCarloGraphSearchParameters>(m_arena.algorithm());
        const std::uint32_t completedChildVisits =
            completedVisits(child.visits, child.virtual_loss);
        const std::uint32_t completedEdgeVisits = completedVisits(edge.visits, edge.virtual_loss);
        if (materialized.transposition) {
            GraphSearchStatistics &statistics = m_arena.graphStatistics();
            ++statistics.transposition_traversals;
            statistics.shared_node_visits_observed += completedChildVisits;
            statistics.incoming_edge_visits_observed += completedEdgeVisits;
            const std::uint64_t advantage = completedChildVisits > completedEdgeVisits
                                                ? completedChildVisits - completedEdgeVisits
                                                : 0;
            statistics.shared_visit_advantage += advantage;
            statistics.maximum_shared_visit_advantage =
                std::max(statistics.maximum_shared_visit_advantage, advantage);
        }
        if (materialized.transposition && completedChildVisits > 0 && completedEdgeVisits == 0 &&
            !Game::isTerminal(child.position)) {
            ++m_arena.graphStatistics().evaluations_avoided;
            return GraphSearchSelection{.path = path,
                                        .immediate_value = graphCorrection(edge, child),
                                        .update_leaf = false};
        }
        if (materialized.transposition && completedChildVisits > 0 && completedEdgeVisits > 0) {
            const float childMean = child.value_sum / static_cast<float>(completedChildVisits);
            const float edgeMean = edge.value_sum / static_cast<float>(completedEdgeVisits);
            if (std::abs(childMean - edgeMean) > graphParameters.transposition_value_threshold) {
                ++m_arena.graphStatistics().evaluations_avoided;
                ++m_arena.graphStatistics().transposition_corrections;
                return GraphSearchSelection{.path = path,
                                            .immediate_value = graphCorrection(edge, child),
                                            .update_leaf = false};
            }
            ++m_arena.graphStatistics().continued_transpositions;
        }
        if (Game::isTerminal(child.position)) {
            return GraphSearchSelection{.path = path,
                                        .immediate_value = Game::terminalValue(child.position),
                                        .update_leaf = true};
        }
        trajectoryNodes.push_back(materialized.node_index);
        std::optional<GraphSearchSelection> selected =
            selectAvailableGraphLeaf(materialized.node_index, parameters, path, trajectoryNodes);
        if (selected.has_value()) {
            return selected;
        }
        trajectoryNodes.pop_back();
        path.steps.pop_back();
        path.leaf_index = nodeIndex;
        return std::nullopt;
    }

    [[nodiscard]] float edgeValueForParent(const Node &parent, const Edge &edge,
                                           const TreeSearchParameters &parameters) const {
        if (edge.visits > 0) {
            return (-m_valueDiscountPerPly * edge.value_sum - edge.virtual_loss) /
                   static_cast<float>(edge.visits);
        }
        const float parentMean =
            parent.visits == 0 ? 0.0F : parent.value_sum / static_cast<float>(parent.visits);
        return parameters.first_play_urgency.unvisitedParentValue(parentMean);
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
            value = -value * m_valueDiscountPerPly;
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
            value = -value * m_valueDiscountPerPly;
            nodeIndex = *selected.parent_index;
        }
    }

    [[nodiscard]] static std::uint32_t completedVisits(const std::uint32_t visits,
                                                       const float virtualLoss) {
        const auto reservations = static_cast<std::uint32_t>(std::lround(virtualLoss));
        assert(reservations <= visits);
        return visits - reservations;
    }

    [[nodiscard]] float graphCorrection(const Edge &edge, const Node &child) {
        const std::uint32_t completedChildVisits =
            completedVisits(child.visits, child.virtual_loss);
        assert(completedChildVisits > 0);
        const float target = child.value_sum / static_cast<float>(completedChildVisits);
        const float resultingVisits =
            static_cast<float>(completedVisits(edge.visits, edge.virtual_loss) + 1U);
        const float correction = resultingVisits * target - edge.value_sum;
        const float clipped = std::clamp(correction, -1.0F, 1.0F);
        if (clipped != correction) {
            ++m_arena.graphStatistics().correction_clips;
        }
        return clipped;
    }

    void discountGraph(const float retainedFraction) {
        const auto retain = [retainedFraction](std::uint32_t &visits, float &valueSum) {
            const std::uint32_t originalVisits = visits;
            visits = static_cast<std::uint32_t>(static_cast<double>(visits) * retainedFraction);
            valueSum = search_tree_detail::retainValueSum(valueSum, originalVisits, visits);
        };
        for (const auto index : range(m_arena.capacity())) {
            if (!m_arena.isLive(index)) {
                continue;
            }
            Node &selected = node(index);
            retain(selected.visits, selected.value_sum);
            for (Edge &edge : selected.children) {
                retain(edge.visits, edge.value_sum);
            }
        }
        m_arena.pruneToLiveNodeLimit(
            std::max<std::size_t>(1, static_cast<std::size_t>(root().visits) + 1));
    }
};

template <SearchGame Game> class GameSearchRoot {
public:
    using Position = typename Game::State;
    using Tree = GameSearchTree<Game>;

    GameSearchRoot(Position position, const std::size_t initialCapacity,
                   const std::size_t maximumCapacity = 0, const float valueDiscountPerPly = 1.0F,
                   SearchAlgorithmParameters algorithm = MonteCarloTreeSearchParameters{})
        : m_tree(std::make_shared<Tree>(std::move(position), initialCapacity, maximumCapacity,
                                        valueDiscountPerPly, std::move(algorithm))) {}

    [[nodiscard]] const Position &position() const { return m_tree->root().position; }
    [[nodiscard]] bool isTerminal() const { return Game::isTerminal(position()); }
    [[nodiscard]] std::uint32_t visits() const { return m_tree->root().visits; }
    [[nodiscard]] std::size_t liveNodeCount() const { return m_tree->liveNodeCount(); }
    [[nodiscard]] Tree &tree() { return *m_tree; }
    [[nodiscard]] const Tree &tree() const { return *m_tree; }
    [[nodiscard]] std::shared_ptr<Tree> sharedTree() const { return m_tree; }
    [[nodiscard]] const GraphSearchStatistics &graphStatistics() const {
        return m_tree->graphStatistics();
    }
    [[nodiscard]] GraphStructureStatistics graphStructureStatistics() const {
        return m_tree->graphStructureStatistics();
    }
    void discount(const float retainedFraction) { m_tree->discount(retainedFraction); }
    void play(const int actionId) { m_tree->reroot(actionId); }
    void reset() { m_tree->reset(); }

private:
    std::shared_ptr<Tree> m_tree;
};
