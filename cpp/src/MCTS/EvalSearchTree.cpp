#include "EvalSearchTree.hpp"

#include "MoveEncoding.hpp"

namespace {
constexpr std::uint32_t VIRTUAL_LOSS_DELTA = 1;
constexpr std::uint32_t INVALID_EDGE_INDEX = std::numeric_limits<std::uint32_t>::max();
} // namespace

EvalSearchTree::EvalSearchTree(Board rootBoard, const std::size_t initialCapacity,
                               const EvalEdgeStorage edgeStorage, const std::size_t maximumCapacity)
    : m_edgeResource(edgeStorage == EvalEdgeStorage::UnsynchronizedPool
                         ? static_cast<std::pmr::memory_resource *>(&m_edgePool)
                         : std::pmr::new_delete_resource()),
      m_edgeStorage(edgeStorage), m_slots(initialCapacity), m_maximumCapacity(maximumCapacity) {
    if (initialCapacity == 0 || maximumCapacity < initialCapacity ||
        maximumCapacity > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("EvalSearchTree initial capacity is out of range");
    }
    m_freeSlots.reserve(initialCapacity);
    for (std::size_t slot = initialCapacity; slot > 0; --slot) {
        m_freeSlots.push_back(static_cast<std::uint32_t>(slot - 1));
    }
    m_rootIndex = allocateNode(std::move(rootBoard), INVALID_EVAL_NODE_INDEX, INVALID_EDGE_INDEX);
}

EvalNodeIndex EvalSearchTree::makeIndex(const std::uint32_t slot, const std::uint32_t generation) {
    return (static_cast<EvalNodeIndex>(generation) << 32U) | slot;
}

std::uint32_t EvalSearchTree::slotIndex(const EvalNodeIndex index) {
    return static_cast<std::uint32_t>(index);
}

std::uint32_t EvalSearchTree::generation(const EvalNodeIndex index) {
    return static_cast<std::uint32_t>(index >> 32U);
}

void EvalSearchTree::grow() {
    const std::size_t oldCapacity = m_slots.size();
    if (oldCapacity == m_maximumCapacity) {
        throw std::overflow_error("EvalSearchTree reached its configured node limit");
    }
    const std::size_t newCapacity = std::min(m_maximumCapacity, oldCapacity * 2);
    m_slots.resize(newCapacity);
    m_freeSlots.reserve(newCapacity);
    for (std::size_t slot = newCapacity; slot > oldCapacity; --slot) {
        m_freeSlots.push_back(static_cast<std::uint32_t>(slot - 1));
    }
}

EvalNodeIndex EvalSearchTree::allocateNode(Board board, const EvalNodeIndex parent,
                                           const std::uint32_t parentEdge) {
    if (m_freeSlots.empty()) {
        grow();
    }
    const std::uint32_t freeSlot = m_freeSlots.back();
    m_freeSlots.pop_back();
    NodeSlot &slot = m_slots[freeSlot];
    assert(!slot.value.has_value());
    slot.value.emplace(std::move(board), parent, parentEdge, m_edgeStorage, m_edgeResource);
    ++m_liveNodeCount;
    m_maximumLiveNodeCount = std::max(m_maximumLiveNodeCount, m_liveNodeCount);
    return makeIndex(freeSlot, slot.generation);
}

EvalSearchNode &EvalSearchTree::node(const EvalNodeIndex index) {
    if (index == INVALID_EVAL_NODE_INDEX) {
        throw std::logic_error("Invalid EvalSearchTree node index");
    }
    const std::uint32_t requestedSlot = slotIndex(index);
    if (requestedSlot >= m_slots.size()) {
        throw std::out_of_range("EvalSearchTree node slot is outside the arena");
    }
    NodeSlot &slot = m_slots[requestedSlot];
    if (!slot.value.has_value() || slot.generation != generation(index)) {
        throw std::logic_error("Stale EvalSearchTree node index");
    }
    return *slot.value;
}

const EvalSearchNode &EvalSearchTree::node(const EvalNodeIndex index) const {
    return const_cast<EvalSearchTree *>(this)->node(index);
}

EvalSearchEdge &EvalSearchTree::edge(const EvalNodeIndex parentIndex,
                                     const std::uint32_t edgeIndex) {
    EvalSearchNode &parent = node(parentIndex);
    if (edgeIndex >= parent.children.size()) {
        throw std::out_of_range("EvalSearchTree edge index is outside the node");
    }
    return parent.children[edgeIndex];
}

const EvalSearchEdge &EvalSearchTree::edge(const EvalNodeIndex parentIndex,
                                           const std::uint32_t edgeIndex) const {
    return const_cast<EvalSearchTree *>(this)->edge(parentIndex, edgeIndex);
}

void EvalSearchTree::expand(const EvalNodeIndex nodeIndex, const std::vector<MoveScore> &moves,
                            const std::optional<WdlPrediction> outcome) {
    EvalSearchNode &expandedNode = node(nodeIndex);
    if (expandedNode.expanded) {
        return;
    }
    if (moves.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("EvalSearchTree node has too many legal moves");
    }
    expandedNode.network_outcome = outcome;
    expandedNode.children.reserve(moves.size());
    for (const auto &[move, policy] : moves) {
        expandedNode.children.emplaceBack(move, policy);
    }
    expandedNode.expanded = true;
}

EvalNodeIndex EvalSearchTree::materializeChild(const EvalNodeIndex parentIndex,
                                               const std::uint32_t edgeIndex) {
    const EvalNodeIndex existingChild = edge(parentIndex, edgeIndex).child;
    if (existingChild != INVALID_EVAL_NODE_INDEX) {
        static_cast<void>(node(existingChild));
        return existingChild;
    }
    Board childBoard = node(parentIndex).board;
    childBoard.makeMove(edge(parentIndex, edgeIndex).move);
    const EvalNodeIndex childIndex = allocateNode(std::move(childBoard), parentIndex, edgeIndex);
    edge(parentIndex, edgeIndex).child = childIndex;
    return childIndex;
}

EvalSearchStatistics &EvalSearchTree::statistics(const EvalNodeIndex index) {
    if (index == m_rootIndex) {
        return m_rootStatistics;
    }
    const EvalSearchNode &selectedNode = node(index);
    return edge(selectedNode.parent, selectedNode.parent_edge).statistics;
}

const EvalSearchStatistics &EvalSearchTree::statistics(const EvalNodeIndex index) const {
    return const_cast<EvalSearchTree *>(this)->statistics(index);
}

std::uint32_t EvalSearchTree::bestChildEdge(const EvalNodeIndex parentIndex,
                                            const float explorationConstant) const {
    const EvalSearchNode &parent = node(parentIndex);
    if (!parent.expanded || parent.children.empty()) {
        throw std::logic_error("Cannot select an edge from a leaf node");
    }
    const float explorationCommon =
        explorationConstant * std::sqrt(static_cast<float>(statistics(parentIndex).visits));
    float bestValue = -std::numeric_limits<float>::infinity();
    std::uint32_t bestIndex = 0;
    for (std::uint32_t index = 0; index < parent.children.size(); ++index) {
        const EvalSearchEdge &candidate = parent.children[index];
        const float exploration = candidate.policy * explorationCommon /
                                  static_cast<float>(1U + candidate.statistics.visits);
        float actionValue = 0.0F;
        if (candidate.statistics.visits > 0) {
            actionValue = -(candidate.statistics.result_sum +
                            static_cast<float>(candidate.statistics.virtual_loss)) /
                          static_cast<float>(candidate.statistics.visits);
        }
        const float value = exploration + actionValue;
        if (value > bestValue) {
            bestValue = value;
            bestIndex = index;
        }
    }
    return bestIndex;
}

std::uint32_t EvalSearchTree::preferredRootEdge() const {
    const EvalSearchNode &root = node(m_rootIndex);
    if (root.children.empty()) {
        throw std::logic_error("Cannot choose a move from an unexpanded EvalSearchTree root");
    }
    std::uint32_t preferred = 0;
    for (std::uint32_t index = 1; index < root.children.size(); ++index) {
        const EvalSearchEdge &candidate = root.children[index];
        const EvalSearchEdge &current = root.children[preferred];
        if (candidate.statistics.visits > current.statistics.visits ||
            (candidate.statistics.visits == current.statistics.visits &&
             (candidate.policy > current.policy ||
              (candidate.policy == current.policy &&
               toString(candidate.move) < toString(current.move))))) {
            preferred = index;
        }
    }
    return preferred;
}

EvalNodeIndex EvalSearchTree::selectLeaf(const float explorationConstant) {
    EvalNodeIndex selected = m_rootIndex;
    while (node(selected).expanded && !node(selected).children.empty()) {
        selected = materializeChild(selected, bestChildEdge(selected, explorationConstant));
    }
    return selected;
}

void EvalSearchTree::addVirtualLoss(EvalNodeIndex leafIndex) {
    while (true) {
        EvalSearchStatistics &selectedStatistics = statistics(leafIndex);
        ++selectedStatistics.virtual_loss;
        ++selectedStatistics.visits;
        if (leafIndex == m_rootIndex) {
            return;
        }
        leafIndex = node(leafIndex).parent;
    }
}

void EvalSearchTree::cancelVirtualLoss(EvalNodeIndex leafIndex) {
    while (true) {
        EvalSearchStatistics &selectedStatistics = statistics(leafIndex);
        assert(selectedStatistics.virtual_loss >= VIRTUAL_LOSS_DELTA);
        assert(selectedStatistics.visits >= VIRTUAL_LOSS_DELTA);
        selectedStatistics.virtual_loss -= VIRTUAL_LOSS_DELTA;
        selectedStatistics.visits -= VIRTUAL_LOSS_DELTA;
        if (leafIndex == m_rootIndex) {
            return;
        }
        leafIndex = node(leafIndex).parent;
    }
}

void EvalSearchTree::backPropagate(EvalNodeIndex leafIndex, float result) {
    while (true) {
        EvalSearchStatistics &selectedStatistics = statistics(leafIndex);
        selectedStatistics.result_sum += result;
        ++selectedStatistics.visits;
        if (leafIndex == m_rootIndex) {
            return;
        }
        leafIndex = node(leafIndex).parent;
        result = -result;
    }
}

void EvalSearchTree::backPropagateAndRemoveVirtualLoss(EvalNodeIndex leafIndex, float result) {
    while (true) {
        EvalSearchStatistics &selectedStatistics = statistics(leafIndex);
        assert(selectedStatistics.virtual_loss >= VIRTUAL_LOSS_DELTA);
        selectedStatistics.result_sum += result;
        selectedStatistics.virtual_loss -= VIRTUAL_LOSS_DELTA;
        if (leafIndex == m_rootIndex) {
            return;
        }
        leafIndex = node(leafIndex).parent;
        result = -result;
    }
}

void EvalSearchTree::releaseNode(const EvalNodeIndex index) {
    if (index == m_rootIndex) {
        throw std::logic_error("Cannot release the active EvalSearchTree root");
    }
    NodeSlot &slot = m_slots[slotIndex(index)];
    static_cast<void>(node(index));
    slot.value.reset();
    if (slot.generation == std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("EvalSearchTree node generation exhausted");
    }
    ++slot.generation;
    m_freeSlots.push_back(slotIndex(index));
    --m_liveNodeCount;
}

void EvalSearchTree::reclaimSubtree(const EvalNodeIndex index) {
    std::vector<std::pair<EvalNodeIndex, bool>> pending{{index, false}};
    while (!pending.empty()) {
        const auto [current, visited] = pending.back();
        pending.pop_back();
        if (visited) {
            releaseNode(current);
            continue;
        }
        pending.emplace_back(current, true);
        for (const EvalSearchEdge &descendant : node(current).children.edges()) {
            if (descendant.child != INVALID_EVAL_NODE_INDEX) {
                pending.emplace_back(descendant.child, false);
            }
        }
    }
}

EvalNodeIndex EvalSearchTree::reroot(const std::uint32_t edgeIndex) {
    const EvalNodeIndex oldRootIndex = m_rootIndex;
    const EvalNodeIndex retainedIndex = materializeChild(oldRootIndex, edgeIndex);
    const EvalSearchEdge retainedEdge = edge(oldRootIndex, edgeIndex);

    EvalSearchNode &oldRoot = node(oldRootIndex);
    for (std::uint32_t index = 0; index < oldRoot.children.size(); ++index) {
        EvalSearchEdge &discardedEdge = oldRoot.children[index];
        if (index != edgeIndex && discardedEdge.child != INVALID_EVAL_NODE_INDEX) {
            reclaimSubtree(discardedEdge.child);
            discardedEdge.child = INVALID_EVAL_NODE_INDEX;
        }
    }

    edge(oldRootIndex, edgeIndex).child = INVALID_EVAL_NODE_INDEX;
    EvalSearchNode &retainedRoot = node(retainedIndex);
    retainedRoot.parent = INVALID_EVAL_NODE_INDEX;
    retainedRoot.parent_edge = INVALID_EDGE_INDEX;
    m_rootIndex = retainedIndex;
    m_rootStatistics = retainedEdge.statistics;
    releaseNode(oldRootIndex);
    return retainedIndex;
}

int EvalSearchTree::maximumDepth() const {
    int maximumDepth = 1;
    std::vector<std::pair<EvalNodeIndex, int>> pending{{m_rootIndex, 1}};
    while (!pending.empty()) {
        const auto [current, depth] = pending.back();
        pending.pop_back();
        maximumDepth = std::max(maximumDepth, depth);
        for (const EvalSearchEdge &descendant : node(current).children.edges()) {
            if (descendant.child != INVALID_EVAL_NODE_INDEX) {
                pending.emplace_back(descendant.child, depth + 1);
            }
        }
    }
    return maximumDepth;
}

std::size_t EvalSearchTree::totalEdgeCount() const {
    std::size_t total = 0;
    for (const NodeSlot &slot : m_slots) {
        if (slot.value.has_value()) {
            total += slot.value->children.size();
        }
    }
    return total;
}

EvalVisitCounts EvalSearchTree::gatherVisitCounts() const {
    const EvalSearchNode &root = node(m_rootIndex);
    EvalVisitCounts counts;
    counts.reserve(root.children.size());
    for (const EvalSearchEdge &rootEdge : root.children.edges()) {
        counts.emplace_back(encodeMove(rootEdge.move, &root.board),
                            static_cast<int>(rootEdge.statistics.visits));
    }
    return counts;
}
