#pragma once

#include "games/GameConcepts.hpp"
#include "search/tree/TreeSearchParameters.hpp"
#include "search/tree/TreeTypes.hpp"
#include "util/Timing.hpp"
#include "util/py.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <queue>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace search_tree_detail {

struct GraphChildMaterialization {
    std::size_t node_index;
    bool transposition;
    bool cycle;
};

template <SearchGame Game> class TreeArena {
public:
    using Position = typename Game::State;
    using Node = GameSearchNode<Game>;
    using Edge = GameSearchEdge<typename Game::Action>;

    TreeArena(Position rootPosition, const std::size_t initialCapacity,
              const std::size_t maximumCapacity,
              SearchAlgorithmParameters algorithm = MonteCarloTreeSearchParameters{})
        : m_maximumCapacity(maximumCapacity == 0 ? initialCapacity : maximumCapacity),
          m_initialPosition(rootPosition), m_algorithm(std::move(algorithm)) {
        if (initialCapacity == 0 || m_maximumCapacity < initialCapacity) {
            throw std::invalid_argument("Game search tree capacity must be positive");
        }
        m_nodes.reserve(initialCapacity);
        m_nodes.resize(initialCapacity);
        for (const auto offset : range(initialCapacity)) {
            m_freeSlots.push_back(initialCapacity - offset - 1);
        }
        m_rootIndex = allocateNode(std::move(rootPosition), std::nullopt, std::nullopt);
        initializeGraphRoot();
    }

    [[nodiscard]] const Node &node(const std::size_t index) const {
        if (index >= m_nodes.size() || !m_nodes[index].has_value()) {
            throw std::out_of_range("Game search node is not live");
        }
        return *m_nodes[index];
    }

    [[nodiscard]] Node &node(const std::size_t index) {
        return const_cast<Node &>(std::as_const(*this).node(index));
    }

    [[nodiscard]] const Node &root() const { return node(m_rootIndex); }
    [[nodiscard]] Node &root() { return node(m_rootIndex); }
    [[nodiscard]] std::size_t rootIndex() const noexcept { return m_rootIndex; }
    [[nodiscard]] std::size_t liveNodeCount() const noexcept { return m_liveNodeCount; }
    [[nodiscard]] std::size_t capacity() const noexcept { return m_nodes.size(); }
    [[nodiscard]] bool isLive(const std::size_t index) const noexcept {
        return index < m_nodes.size() && m_nodes[index].has_value();
    }

    [[nodiscard]] std::size_t totalChildCount() const noexcept {
        std::size_t count = 0;
        for (const std::optional<Node> &slot : m_nodes) {
            if (slot.has_value()) {
                count += slot->children.size();
            }
        }
        return count;
    }

    [[nodiscard]] bool graphSearch() const noexcept {
        return std::holds_alternative<MonteCarloGraphSearchParameters>(m_algorithm);
    }

    [[nodiscard]] const SearchAlgorithmParameters &algorithm() const noexcept {
        return m_algorithm;
    }

    [[nodiscard]] const GraphSearchStatistics &graphStatistics() const noexcept {
        return m_graphStatistics;
    }

    [[nodiscard]] GraphSearchStatistics &graphStatistics() noexcept { return m_graphStatistics; }

    [[nodiscard]] GraphStructureStatistics graphStructureStatistics() const {
        GraphStructureStatistics result;
        std::vector<bool> reachable(m_nodes.size(), false);
        std::vector<std::size_t> pending = {m_rootIndex};
        while (!pending.empty()) {
            const std::size_t index = pending.back();
            pending.pop_back();
            if (reachable[index]) {
                continue;
            }
            reachable[index] = true;
            ++result.canonical_nodes;
            if (node(index).expanded()) {
                ++result.expanded_nodes;
            }
            for (const Edge &edge : node(index).children) {
                if (edge.child_index.has_value()) {
                    ++result.materialized_edges;
                    pending.push_back(*edge.child_index);
                }
            }
        }

        std::vector<std::size_t> remainingIncoming(m_nodes.size(), 0);
        for (const auto index : range(m_nodes.size())) {
            if (!reachable[index]) {
                continue;
            }
            for (const Edge &edge : node(index).children) {
                if (edge.child_index.has_value() && reachable[*edge.child_index]) {
                    ++remainingIncoming[*edge.child_index];
                }
            }
        }
        for (const auto index : range(m_nodes.size())) {
            if (reachable[index] && remainingIncoming[index] > 1) {
                ++result.shared_nodes;
            }
        }

        const auto saturatingAdd = [&result](const std::uint64_t left,
                                             const std::uint64_t right) {
            if (right > std::numeric_limits<std::uint64_t>::max() - left) {
                result.saturated = true;
                return std::numeric_limits<std::uint64_t>::max();
            }
            return left + right;
        };
        std::vector<std::uint64_t> pathMultiplicity(m_nodes.size(), 0);
        pathMultiplicity[m_rootIndex] = 1;
        std::queue<std::size_t> ready;
        ready.push(m_rootIndex);
        std::uint64_t processed = 0;
        while (!ready.empty()) {
            const std::size_t index = ready.front();
            ready.pop();
            ++processed;
            const std::uint64_t paths = pathMultiplicity[index];
            result.maximum_path_multiplicity =
                std::max(result.maximum_path_multiplicity, paths);
            result.unfolded_tree_nodes = saturatingAdd(result.unfolded_tree_nodes, paths);
            if (node(index).expanded()) {
                result.unfolded_expanded_nodes =
                    saturatingAdd(result.unfolded_expanded_nodes, paths);
            }
            for (const Edge &edge : node(index).children) {
                if (!edge.child_index.has_value()) {
                    continue;
                }
                const std::size_t childIndex = *edge.child_index;
                result.unfolded_tree_edges = saturatingAdd(result.unfolded_tree_edges, paths);
                pathMultiplicity[childIndex] =
                    saturatingAdd(pathMultiplicity[childIndex], paths);
                assert(remainingIncoming[childIndex] > 0);
                --remainingIncoming[childIndex];
                if (remainingIncoming[childIndex] == 0) {
                    ready.push(childIndex);
                }
            }
        }
        if (processed != result.canonical_nodes) {
            throw std::logic_error("Game search graph structure contains a cycle");
        }
        return result;
    }

    void recordEdgesCreated(const std::size_t count) {
        if (!graphSearch()) {
            return;
        }
        m_graphStatistics.edges_created += count;
        m_liveEdgeCount += count;
        updateGraphPeaks();
    }

    [[nodiscard]] std::size_t materializeChild(const std::size_t parentIndex,
                                               const std::size_t edgeIndex) {
        Edge &edge = node(parentIndex).children.at(edgeIndex);
        if (edge.child_index.has_value()) {
            return *edge.child_index;
        }
        const Position child = Game::childState(node(parentIndex).position, edge.action);
        const std::size_t childIndex = allocateNode(child, parentIndex, edgeIndex);
        Edge &retainedEdge = node(parentIndex).children[edgeIndex];
        Node &childNode = node(childIndex);
        childNode.visits = retainedEdge.visits;
        childNode.value_sum = retainedEdge.value_sum;
        retainedEdge.child_index = childIndex;
        return childIndex;
    }

    [[nodiscard]] GraphChildMaterialization
    materializeGraphChild(const std::size_t parentIndex, const std::size_t edgeIndex,
                          const std::vector<std::size_t> &trajectoryNodes) {
        if (!graphSearch()) {
            throw std::logic_error("Graph materialization requires graph-search parameters");
        }
        Edge &edge = node(parentIndex).children.at(edgeIndex);
        if (edge.child_index.has_value()) {
            const bool cycle =
                std::ranges::find(trajectoryNodes, *edge.child_index) != trajectoryNodes.end();
            if (cycle) {
                ++m_graphStatistics.cycle_cutoffs;
            }
            return {.node_index = *edge.child_index,
                    .transposition = node(*edge.child_index).incoming_edges > 1,
                    .cycle = cycle};
        }

        const Position child = Game::childState(node(parentIndex).position, edge.action);
        const auto &parameters = std::get<MonteCarloGraphSearchParameters>(m_algorithm);
        if (parameters.transpositions_enabled) {
            ScopedNanosecondTimer lookupTimer(m_graphStatistics.identity_lookup_nanoseconds);
            ++m_graphStatistics.transposition_table_probes;
            const std::size_t hash = Game::stateHash(child);
            const auto [bucketBegin, bucketEnd] = m_transpositionTable.equal_range(hash);
            if (bucketBegin != bucketEnd) {
                for (auto candidate = bucketBegin; candidate != bucketEnd; ++candidate) {
                    const std::size_t candidateIndex = candidate->second;
                    ++m_graphStatistics.hash_collision_checks;
                    if (!Game::statesEqual(child, node(candidateIndex).position)) {
                        continue;
                    }
                    ++m_graphStatistics.transposition_table_hits;
                    if (std::ranges::find(trajectoryNodes, candidateIndex) !=
                        trajectoryNodes.end()) {
                        ++m_graphStatistics.cycle_cutoffs;
                        return {.node_index = candidateIndex, .transposition = true, .cycle = true};
                    }
                    edge.child_index = candidateIndex;
                    ++node(candidateIndex).incoming_edges;
                    ++m_graphStatistics.transposition_links;
                    return {.node_index = candidateIndex, .transposition = true, .cycle = false};
                }
            }
        }

        const std::size_t childIndex = allocateNode(child, std::nullopt, std::nullopt);
        Node &childNode = node(childIndex);
        childNode.visits = edge.visits;
        childNode.value_sum = edge.value_sum;
        childNode.incoming_edges = 1;
        edge.child_index = childIndex;
        if (parameters.transpositions_enabled) {
            m_transpositionTable.emplace(Game::stateHash(child), childIndex);
        }
        ++m_graphStatistics.unique_nodes_created;
        updateGraphPeaks();
        return {.node_index = childIndex, .transposition = false, .cycle = false};
    }

    void rerootEdge(const std::size_t edgeIndex) {
        if (graphSearch()) {
            rerootGraphEdge(edgeIndex);
            return;
        }
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
        m_liveEdgeCount = 0;
        m_rootIndex = allocateNode(m_initialPosition, std::nullopt, std::nullopt);
        m_transpositionTable.clear();
        initializeGraphRoot();
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
        if (graphSearch()) {
            pruneGraphToLiveNodeLimit(liveNodeLimit);
            return;
        }
        if (m_liveNodeCount <= liveNodeLimit) {
            return;
        }
        requireIdle();
        using PruneCandidate = std::tuple<std::uint32_t, int, std::size_t, std::size_t>;
        std::priority_queue<PruneCandidate, std::vector<PruneCandidate>,
                            std::greater<PruneCandidate>>
            candidates;
        std::vector<std::size_t> materializedChildCounts(m_nodes.size(), 0);
        std::vector<std::size_t> traversalOrders(m_nodes.size(), 0);
        std::vector<std::size_t> pending = {m_rootIndex};
        std::size_t traversalOrder = 0;
        while (!pending.empty()) {
            const std::size_t currentIndex = pending.back();
            pending.pop_back();
            for (const Edge &edge : std::ranges::reverse_view(node(currentIndex).children)) {
                if (edge.child_index.has_value()) {
                    ++materializedChildCounts[currentIndex];
                    traversalOrders[*edge.child_index] = ++traversalOrder;
                    pending.push_back(*edge.child_index);
                }
            }
        }

        for (const auto index : range(m_nodes.size())) {
            if (index == m_rootIndex || !m_nodes[index].has_value() ||
                materializedChildCounts[index] != 0) {
                continue;
            }
            candidates.emplace(node(index).visits, -nodeDepth(index), traversalOrders[index],
                               index);
        }

        while (m_liveNodeCount > liveNodeLimit) {
            if (candidates.empty()) {
                throw std::logic_error("Game search tree has no prunable leaf");
            }
            const auto [visits, negativeDepth, order, index] = candidates.top();
            static_cast<void>(visits);
            static_cast<void>(negativeDepth);
            static_cast<void>(order);
            candidates.pop();
            if (!m_nodes[index].has_value() || materializedChildCounts[index] != 0) {
                continue;
            }
            const Node &pruned = node(index);
            const std::size_t parentIndex = *pruned.parent_index;
            node(parentIndex).children[*pruned.parent_edge_index].child_index.reset();
            releaseNode(index);
            --materializedChildCounts[parentIndex];
            if (parentIndex != m_rootIndex && materializedChildCounts[parentIndex] == 0) {
                candidates.emplace(node(parentIndex).visits, -nodeDepth(parentIndex),
                                   traversalOrders[parentIndex], parentIndex);
            }
        }
    }

    void requireIdle() const {
        for (const std::optional<Node> &slot : m_nodes) {
            if (!slot.has_value()) {
                continue;
            }
            if (slot->inference_pending || slot->virtual_loss != 0.0F) {
                throw std::logic_error("Cannot retain or prune a game search with pending work");
            }
            for (const Edge &edge : slot->children) {
                if (edge.virtual_loss != 0.0F) {
                    throw std::logic_error(
                        "Cannot retain or prune a game search with pending work");
                }
            }
        }
    }

private:
    std::size_t m_maximumCapacity;
    Position m_initialPosition;
    std::vector<std::optional<Node>> m_nodes;
    std::vector<std::size_t> m_freeSlots;
    std::size_t m_liveNodeCount = 0;
    std::size_t m_rootIndex;
    SearchAlgorithmParameters m_algorithm;
    std::unordered_multimap<std::size_t, std::size_t> m_transpositionTable;
    GraphSearchStatistics m_graphStatistics;
    std::size_t m_liveEdgeCount = 0;

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
        if (graphSearch()) {
            m_liveEdgeCount -= node(index).children.size();
        }
        m_nodes[index].reset();
        m_freeSlots.push_back(index);
        --m_liveNodeCount;
    }

    [[nodiscard]] int nodeDepth(std::size_t index) const {
        int depth = 0;
        while (index != m_rootIndex) {
            const Node &selected = node(index);
            index = *selected.parent_index;
            ++depth;
        }
        return depth;
    }

    void initializeGraphRoot() {
        if (!graphSearch()) {
            return;
        }
        m_transpositionTable.emplace(Game::stateHash(root().position), m_rootIndex);
        ++m_graphStatistics.unique_nodes_created;
        updateGraphPeaks();
    }

    void updateGraphPeaks() {
        m_graphStatistics.peak_live_nodes = std::max(m_graphStatistics.peak_live_nodes,
                                                     static_cast<std::uint64_t>(m_liveNodeCount));
        m_graphStatistics.peak_live_edges = std::max(m_graphStatistics.peak_live_edges,
                                                     static_cast<std::uint64_t>(m_liveEdgeCount));
    }

    void rerootGraphEdge(const std::size_t edgeIndex) {
        requireIdle();
        ScopedNanosecondTimer timer(m_graphStatistics.reroot_nanoseconds);
        if (edgeIndex >= root().children.size()) {
            throw std::out_of_range("Cannot reroot to a missing game-search edge");
        }
        const std::vector<std::size_t> rootPath = {m_rootIndex};
        const GraphChildMaterialization retained =
            materializeGraphChild(m_rootIndex, edgeIndex, rootPath);
        if (retained.cycle) {
            throw std::logic_error("Cannot reroot a graph search through a cycle");
        }
        m_rootIndex = retained.node_index;
        m_initialPosition = node(m_rootIndex).position;
        sweepUnreachableFromRoot();
        m_graphStatistics.nodes_retained += m_liveNodeCount;
    }

    void sweepUnreachableFromRoot() {
        std::unordered_set<std::size_t> reachable;
        std::vector<std::size_t> pending = {m_rootIndex};
        while (!pending.empty()) {
            const std::size_t index = pending.back();
            pending.pop_back();
            if (!reachable.insert(index).second) {
                continue;
            }
            for (const Edge &edge : node(index).children) {
                if (edge.child_index.has_value()) {
                    pending.push_back(*edge.child_index);
                }
            }
        }
        std::uint64_t reclaimedEdges = 0;
        for (const auto index : range(m_nodes.size())) {
            if (!m_nodes[index].has_value() || reachable.contains(index)) {
                continue;
            }
            reclaimedEdges += m_nodes[index]->children.size();
            releaseNode(index);
            ++m_graphStatistics.nodes_reclaimed;
        }
        m_graphStatistics.edges_reclaimed += reclaimedEdges;
        rebuildGraphMetadata();
    }

    void rebuildGraphMetadata() {
        m_transpositionTable.clear();
        for (std::optional<Node> &slot : m_nodes) {
            if (slot.has_value()) {
                slot->incoming_edges = 0;
            }
        }
        const auto &parameters = std::get<MonteCarloGraphSearchParameters>(m_algorithm);
        for (const auto index : range(m_nodes.size())) {
            if (!m_nodes[index].has_value()) {
                continue;
            }
            if (parameters.transpositions_enabled) {
                m_transpositionTable.emplace(Game::stateHash(node(index).position), index);
            }
            for (const Edge &edge : node(index).children) {
                if (edge.child_index.has_value()) {
                    ++node(*edge.child_index).incoming_edges;
                }
            }
        }
        updateGraphPeaks();
    }

    void pruneGraphToLiveNodeLimit(const std::size_t liveNodeLimit) {
        if (m_liveNodeCount <= liveNodeLimit) {
            return;
        }
        requireIdle();
        ScopedNanosecondTimer timer(m_graphStatistics.pruning_nanoseconds);
        using IncomingReference = std::pair<std::size_t, std::size_t>;
        std::vector<std::vector<IncomingReference>> incomingReferences(m_nodes.size());
        std::vector<std::size_t> materializedChildCounts(m_nodes.size(), 0);
        for (const auto parentIndex : range(m_nodes.size())) {
            if (!m_nodes[parentIndex].has_value()) {
                continue;
            }
            const std::size_t edgeCount = node(parentIndex).children.size();
            for (const auto edgeIndex : range(edgeCount)) {
                const Edge &edge = node(parentIndex).children[edgeIndex];
                if (!edge.child_index.has_value()) {
                    continue;
                }
                ++materializedChildCounts[parentIndex];
                incomingReferences[*edge.child_index].emplace_back(parentIndex, edgeIndex);
            }
        }

        using PruneCandidate = std::pair<std::uint32_t, std::size_t>;
        std::priority_queue<PruneCandidate, std::vector<PruneCandidate>,
                            std::greater<PruneCandidate>>
            candidates;
        for (const auto index : range(m_nodes.size())) {
            if (index != m_rootIndex && m_nodes[index].has_value() &&
                materializedChildCounts[index] == 0) {
                candidates.emplace(node(index).visits, index);
            }
        }

        while (m_liveNodeCount > liveNodeLimit) {
            if (candidates.empty()) {
                throw std::logic_error("Game search graph has no prunable leaf");
            }
            const auto [visits, candidateIndex] = candidates.top();
            static_cast<void>(visits);
            candidates.pop();
            if (!m_nodes[candidateIndex].has_value() ||
                materializedChildCounts[candidateIndex] != 0) {
                continue;
            }
            for (const auto [parentIndex, edgeIndex] : incomingReferences[candidateIndex]) {
                if (!m_nodes[parentIndex].has_value()) {
                    continue;
                }
                Edge &incomingEdge = node(parentIndex).children[edgeIndex];
                if (incomingEdge.child_index != candidateIndex) {
                    continue;
                }
                incomingEdge.child_index.reset();
                --materializedChildCounts[parentIndex];
                if (parentIndex != m_rootIndex && materializedChildCounts[parentIndex] == 0) {
                    candidates.emplace(node(parentIndex).visits, parentIndex);
                }
            }
            releaseNode(candidateIndex);
            ++m_graphStatistics.nodes_pruned;
        }
        rebuildGraphMetadata();
    }
};

} // namespace search_tree_detail
