#pragma once

#include "games/GameConcepts.hpp"
#include "search/tree/TreeTypes.hpp"
#include "util/py.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>
#include <queue>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace search_tree_detail {

template <SearchGame Game> class TreeArena {
public:
    using Position = typename Game::State;
    using Node = GameSearchNode<Game>;
    using Edge = GameSearchEdge<typename Game::Action>;

    TreeArena(Position rootPosition, const std::size_t initialCapacity,
              const std::size_t maximumCapacity)
        : m_maximumCapacity(maximumCapacity == 0 ? initialCapacity : maximumCapacity),
          m_initialPosition(rootPosition) {
        if (initialCapacity == 0 || m_maximumCapacity < initialCapacity) {
            throw std::invalid_argument("Game search tree capacity must be positive");
        }
        m_nodes.reserve(initialCapacity);
        m_nodes.resize(initialCapacity);
        m_positions.resize(initialCapacity);
        for (const auto offset : range(initialCapacity)) {
            m_freeSlots.push_back(initialCapacity - offset - 1);
        }
        m_rootIndex = allocateNode(std::move(rootPosition), std::nullopt, std::nullopt);
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

    // Selection, backup and reroot walk indices they just read out of a live edge or parent link,
    // so the liveness check on that path only costs a branch per node touched.
    [[nodiscard]] const Node &liveNode(const std::size_t index) const noexcept {
        return *m_nodes[index];
    }
    [[nodiscard]] Node &liveNode(const std::size_t index) noexcept { return *m_nodes[index]; }

    [[nodiscard]] const Position &position(const std::size_t index) const {
        if (index >= m_positions.size() || !m_positions[index].has_value()) {
            throw std::out_of_range("Game search position is not live");
        }
        return *m_positions[index];
    }
    [[nodiscard]] const Position &rootPosition() const { return position(m_rootIndex); }

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

    [[nodiscard]] std::size_t materializeChild(const std::size_t parentIndex,
                                               const std::size_t edgeIndex) {
        Edge &edge = node(parentIndex).children.at(edgeIndex);
        if (edge.child_index.has_value()) {
            return *edge.child_index;
        }
        Position child = Game::childState(position(parentIndex), edge.action);
        const std::size_t childIndex = allocateNode(std::move(child), parentIndex, edgeIndex);
        Edge &retainedEdge = node(parentIndex).children[edgeIndex];
        Node &childNode = node(childIndex);
        childNode.visits = retainedEdge.visits;
        childNode.value_sum = retainedEdge.value_sum;
        retainedEdge.child_index = childIndex;
        return childIndex;
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
        m_initialPosition = position(retainedIndex);
        releaseNode(oldRootIndex);
    }

    void reset() {
        m_freeSlots.clear();
        for (const auto offset : range(m_nodes.size())) {
            const std::size_t slot = m_nodes.size() - offset - 1;
            m_nodes[slot].reset();
            m_positions[slot].reset();
            m_freeSlots.push_back(slot);
        }
        m_liveNodeCount = 0;
        m_rootIndex = allocateNode(m_initialPosition, std::nullopt, std::nullopt);
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
    std::vector<std::optional<Position>> m_positions;
    std::vector<std::size_t> m_freeSlots;
    std::size_t m_liveNodeCount = 0;
    std::size_t m_rootIndex;

    [[nodiscard]] std::size_t allocateNode(Position position,
                                           const std::optional<std::size_t> parentIndex,
                                           const std::optional<std::size_t> parentEdgeIndex) {
        if (m_freeSlots.empty() && m_nodes.size() < m_maximumCapacity) {
            const std::size_t oldCapacity = m_nodes.size();
            const std::size_t newCapacity =
                std::min(m_maximumCapacity, std::max(oldCapacity + 1, oldCapacity * 2));
            m_nodes.resize(newCapacity);
            m_positions.resize(newCapacity);
            for (const auto offset : range(newCapacity - oldCapacity)) {
                m_freeSlots.push_back(newCapacity - offset - 1);
            }
        }
        if (m_freeSlots.empty()) {
            throw std::overflow_error("Game search tree capacity exhausted");
        }
        const std::size_t slot = m_freeSlots.back();
        m_freeSlots.pop_back();
        m_positions[slot].emplace(std::move(position));
        m_nodes[slot].emplace(Node{
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
        m_positions[index].reset();
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
};

} // namespace search_tree_detail
