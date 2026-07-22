#include "SearchTree.hpp"

#include "MoveEncoding.hpp"

namespace {
constexpr uint32 VIRTUAL_LOSS_DELTA = 1;
constexpr float TURN_DISCOUNT = 0.99f;
constexpr uint32 INVALID_CHILD_POSITION = std::numeric_limits<uint32>::max();
} // namespace

SearchTree::SearchTree(Board rootBoard, const uint32 capacity) : m_slots(capacity) {
    if (capacity == 0) {
        throw std::invalid_argument("SearchTree capacity must be positive");
    }

    m_freeSlots.reserve(capacity);
    for (uint32 slot = capacity; slot > 0; --slot) {
        m_freeSlots.push_back(slot - 1);
    }
    m_rootIndex = allocateNode(std::move(rootBoard), INVALID_NODE_INDEX, INVALID_CHILD_POSITION);
}

NodeIndex SearchTree::makeIndex(const uint32 slotIndex, const uint32 generation) {
    return (static_cast<NodeIndex>(generation) << 32U) | static_cast<NodeIndex>(slotIndex);
}

uint32 SearchTree::slotIndex(const NodeIndex index) { return static_cast<uint32>(index); }

uint32 SearchTree::generation(const NodeIndex index) { return static_cast<uint32>(index >> 32U); }

NodeIndex SearchTree::allocateNode(Board board, const NodeIndex parentIndex,
                                   const uint32 parentChildIndex) {
    if (m_freeSlots.empty()) {
        throw std::overflow_error(
            "SearchTree arena exhausted at " + std::to_string(m_liveNodeCount) + "/" +
            std::to_string(capacity()) + " live nodes with " +
            std::to_string(m_rootStatistics.number_of_visits) + " root visits");
    }

    const uint32 freeSlot = m_freeSlots.back();
    m_freeSlots.pop_back();
    NodeSlot &slot = m_slots[freeSlot];
    if (slot.value.has_value()) {
        throw std::logic_error("SearchTree free list references a live node");
    }
    slot.value.emplace(std::move(board), parentIndex, parentChildIndex);
    ++m_liveNodeCount;
    return makeIndex(freeSlot, slot.generation);
}

SearchNode &SearchTree::node(const NodeIndex index) {
    if (index == INVALID_NODE_INDEX) {
        throw std::logic_error("Invalid SearchTree node index");
    }
    const uint32 requestedSlot = slotIndex(index);
    if (requestedSlot >= m_slots.size()) {
        throw std::out_of_range("SearchTree node slot is outside the arena");
    }

    NodeSlot &slot = m_slots[requestedSlot];
    if (!slot.value.has_value() || slot.generation != generation(index)) {
        throw std::logic_error("Stale SearchTree node index");
    }
    return *slot.value;
}

const SearchNode &SearchTree::node(const NodeIndex index) const {
    return const_cast<SearchTree *>(this)->node(index);
}

Child &SearchTree::child(const NodeIndex parentIndex, const uint32 childIndex) {
    SearchNode &parent = node(parentIndex);
    if (childIndex >= parent.children.size()) {
        throw std::out_of_range("SearchTree child index is outside the node");
    }
    return parent.children[childIndex];
}

const Child &SearchTree::child(const NodeIndex parentIndex, const uint32 childIndex) const {
    return const_cast<SearchTree *>(this)->child(parentIndex, childIndex);
}

void SearchTree::releaseNode(const NodeIndex index) {
    SearchNode &releasedNode = node(index);
    if (index == m_rootIndex) {
        throw std::logic_error("Cannot directly release the active SearchTree root");
    }
    static_cast<void>(releasedNode);

    const uint32 releasedSlot = slotIndex(index);
    NodeSlot &slot = m_slots[releasedSlot];
    slot.value.reset();
    if (slot.generation == std::numeric_limits<uint32>::max()) {
        throw std::overflow_error("SearchTree node generation exhausted");
    }
    ++slot.generation;
    m_freeSlots.push_back(releasedSlot);
    --m_liveNodeCount;
}

void SearchTree::expand(const NodeIndex nodeIndex, const std::vector<MoveScore> &movesWithScores) {
    TIMEIT("SearchTree::expand");
    SearchNode &expandedNode = node(nodeIndex);
    if (expandedNode.isExpanded() || movesWithScores.empty()) {
        return;
    }
    if (movesWithScores.size() > std::numeric_limits<uint32>::max()) {
        throw std::overflow_error("SearchTree node has too many legal moves");
    }

    expandedNode.children.reserve(movesWithScores.size());
    for (const auto &[move, policy] : movesWithScores) {
        expandedNode.children.emplace_back(move, policy);
    }
}

NodeIndex SearchTree::materializeChild(const NodeIndex parentIndex, const uint32 childIndex) {
    Child &selectedChild = child(parentIndex, childIndex);
    if (selectedChild.node_index != INVALID_NODE_INDEX) {
        static_cast<void>(node(selectedChild.node_index));
        return selectedChild.node_index;
    }

    Board childBoard = node(parentIndex).board;
    childBoard.makeMove(selectedChild.move);
    const NodeIndex materializedIndex =
        allocateNode(std::move(childBoard), parentIndex, childIndex);
    selectedChild.node_index = materializedIndex;
    return materializedIndex;
}

RootStatistics SearchTree::nodeStatistics(const NodeIndex index) const {
    if (index == m_rootIndex) {
        return m_rootStatistics;
    }
    const SearchNode &selectedNode = node(index);
    const Child &incomingEdge = child(selectedNode.parent_index, selectedNode.parent_child_index);
    return {incomingEdge.number_of_visits, incomingEdge.result_sum, incomingEdge.virtual_loss};
}

uint32 SearchTree::bestChildIndex(const NodeIndex parentIndex, const float cParam) const {
    TIMEIT("SearchTree::bestChildIndex");
    const SearchNode &parent = node(parentIndex);
    if (parent.children.empty()) {
        throw std::logic_error("Cannot select a child from an unexpanded node");
    }

    const RootStatistics parentStatistics = nodeStatistics(parentIndex);
    const float explorationCommon =
        cParam * std::sqrt(static_cast<float>(parentStatistics.number_of_visits));
    float bestValue = -std::numeric_limits<float>::infinity();
    uint32 bestIndex = 0;

    for (uint32 index = 0; index < parent.children.size(); ++index) {
        const Child &candidate = parent.children[index];
        const float exploration = candidate.policy * explorationCommon /
                                  static_cast<float>(1U + candidate.number_of_visits);
        float actionValue = 0.0f;
        if (candidate.number_of_visits > 0) {
            actionValue = -(candidate.result_sum + candidate.virtual_loss) /
                          static_cast<float>(candidate.number_of_visits);
        }
        const float value = exploration + actionValue;
        if (value > bestValue) {
            bestValue = value;
            bestIndex = index;
        }
    }
    return bestIndex;
}

void SearchTree::addVirtualLoss(NodeIndex leafIndex) {
    TIMEIT("SearchTree::addVirtualLoss");
    while (leafIndex != m_rootIndex) {
        const SearchNode &selectedNode = node(leafIndex);
        Child &incomingEdge = child(selectedNode.parent_index, selectedNode.parent_child_index);
        incomingEdge.virtual_loss += static_cast<float>(VIRTUAL_LOSS_DELTA);
        ++incomingEdge.number_of_visits;
        leafIndex = selectedNode.parent_index;
    }
    m_rootStatistics.virtual_loss += static_cast<float>(VIRTUAL_LOSS_DELTA);
    ++m_rootStatistics.number_of_visits;
}

void SearchTree::removeVirtualLoss(NodeIndex leafIndex) {
    while (leafIndex != m_rootIndex) {
        const SearchNode &selectedNode = node(leafIndex);
        Child &incomingEdge = child(selectedNode.parent_index, selectedNode.parent_child_index);
        assert(incomingEdge.virtual_loss >= static_cast<float>(VIRTUAL_LOSS_DELTA));
        assert(incomingEdge.number_of_visits >= VIRTUAL_LOSS_DELTA);
        incomingEdge.virtual_loss -= static_cast<float>(VIRTUAL_LOSS_DELTA);
        incomingEdge.number_of_visits -= VIRTUAL_LOSS_DELTA;
        leafIndex = selectedNode.parent_index;
    }
    assert(m_rootStatistics.virtual_loss >= static_cast<float>(VIRTUAL_LOSS_DELTA));
    assert(m_rootStatistics.number_of_visits >= VIRTUAL_LOSS_DELTA);
    m_rootStatistics.virtual_loss -= static_cast<float>(VIRTUAL_LOSS_DELTA);
    m_rootStatistics.number_of_visits -= VIRTUAL_LOSS_DELTA;
}

void SearchTree::backPropagate(NodeIndex leafIndex, float result) {
    TIMEIT("SearchTree::backPropagate");
    while (leafIndex != m_rootIndex) {
        const SearchNode &selectedNode = node(leafIndex);
        Child &incomingEdge = child(selectedNode.parent_index, selectedNode.parent_child_index);
        incomingEdge.result_sum += result;
        ++incomingEdge.number_of_visits;
        result = -result * TURN_DISCOUNT;
        leafIndex = selectedNode.parent_index;
    }
    m_rootStatistics.result_sum += result;
    ++m_rootStatistics.number_of_visits;
}

void SearchTree::backPropagateAndRemoveVirtualLoss(NodeIndex leafIndex, float result) {
    TIMEIT("SearchTree::backPropagateAndRemoveVirtualLoss");
    while (leafIndex != m_rootIndex) {
        const SearchNode &selectedNode = node(leafIndex);
        Child &incomingEdge = child(selectedNode.parent_index, selectedNode.parent_child_index);
        incomingEdge.result_sum += result;
        incomingEdge.virtual_loss -= static_cast<float>(VIRTUAL_LOSS_DELTA);
        result = -result * TURN_DISCOUNT;
        leafIndex = selectedNode.parent_index;
    }
    m_rootStatistics.result_sum += result;
    m_rootStatistics.virtual_loss -= static_cast<float>(VIRTUAL_LOSS_DELTA);
}

void SearchTree::reclaimSubtree(const NodeIndex index) {
    std::vector<std::pair<NodeIndex, bool>> pending;
    pending.emplace_back(index, false);
    while (!pending.empty()) {
        const auto [currentIndex, visited] = pending.back();
        pending.pop_back();
        if (visited) {
            releaseNode(currentIndex);
            continue;
        }

        pending.emplace_back(currentIndex, true);
        for (const Child &descendant : node(currentIndex).children) {
            if (descendant.node_index != INVALID_NODE_INDEX) {
                pending.emplace_back(descendant.node_index, false);
            }
        }
    }
}

NodeIndex SearchTree::reroot(const uint32 childIndex) {
    SearchNode &oldRoot = node(m_rootIndex);
    if (childIndex >= oldRoot.children.size()) {
        throw std::out_of_range("Cannot reroot to a missing child");
    }

    const NodeIndex oldRootIndex = m_rootIndex;
    const NodeIndex retainedIndex = materializeChild(oldRootIndex, childIndex);
    const Child retainedEdge = child(oldRootIndex, childIndex);

    for (uint32 index = 0; index < oldRoot.children.size(); ++index) {
        Child &discardedEdge = oldRoot.children[index];
        if (index != childIndex && discardedEdge.node_index != INVALID_NODE_INDEX) {
            reclaimSubtree(discardedEdge.node_index);
            discardedEdge.node_index = INVALID_NODE_INDEX;
        }
    }

    child(oldRootIndex, childIndex).node_index = INVALID_NODE_INDEX;
    SearchNode &retainedRoot = node(retainedIndex);
    retainedRoot.parent_index = INVALID_NODE_INDEX;
    retainedRoot.parent_child_index = INVALID_CHILD_POSITION;

    m_rootIndex = retainedIndex;
    m_rootStatistics = {retainedEdge.number_of_visits, retainedEdge.result_sum,
                        retainedEdge.virtual_loss};
    m_rootMove = retainedEdge.move;
    releaseNode(oldRootIndex);
    return retainedIndex;
}

void SearchTree::discountStatistics(uint32 &numberOfVisits, float &resultSum,
                                    const float percentageOfNodeVisitsToKeep) {
    numberOfVisits =
        static_cast<uint32>(static_cast<float>(numberOfVisits) * percentageOfNodeVisitsToKeep);
    resultSum = static_cast<float>(static_cast<int>(
        static_cast<float>(static_cast<int>(resultSum)) * percentageOfNodeVisitsToKeep));
    resultSum =
        clamp(resultSum, -static_cast<float>(numberOfVisits), static_cast<float>(numberOfVisits));
}

void SearchTree::pruneToLiveNodeLimit(const uint32 liveNodeLimit) {
    if (m_liveNodeCount <= liveNodeLimit) {
        return;
    }

    std::vector<std::pair<NodeIndex, bool>> pending;
    std::vector<NodeIndex> postOrder;
    pending.emplace_back(m_rootIndex, false);
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
        for (const Child &descendant : node(currentIndex).children) {
            if (descendant.node_index != INVALID_NODE_INDEX) {
                pending.emplace_back(descendant.node_index, false);
            }
        }
    }

    for (const NodeIndex index : postOrder) {
        if (m_liveNodeCount <= liveNodeLimit) {
            break;
        }
        SearchNode &prunedNode = node(index);
        Child &incomingEdge = child(prunedNode.parent_index, prunedNode.parent_child_index);
        incomingEdge.node_index = INVALID_NODE_INDEX;
        releaseNode(index);
    }
}

void SearchTree::discount(const float percentageOfNodeVisitsToKeep) {
    if (percentageOfNodeVisitsToKeep < 0.0f || percentageOfNodeVisitsToKeep > 1.0f) {
        throw std::invalid_argument("SearchTree discount must be in [0, 1]");
    }

    discountStatistics(m_rootStatistics.number_of_visits, m_rootStatistics.result_sum,
                       percentageOfNodeVisitsToKeep);

    std::vector<NodeIndex> pending = {m_rootIndex};
    while (!pending.empty()) {
        const NodeIndex currentIndex = pending.back();
        pending.pop_back();
        SearchNode &currentNode = node(currentIndex);
        for (Child &descendant : currentNode.children) {
            discountStatistics(descendant.number_of_visits, descendant.result_sum,
                               percentageOfNodeVisitsToKeep);
            if (descendant.node_index != INVALID_NODE_INDEX) {
                pending.push_back(descendant.node_index);
            }
        }
    }

    const uint32 liveNodeLimit =
        m_rootStatistics.number_of_visits == std::numeric_limits<uint32>::max()
            ? m_rootStatistics.number_of_visits
            : m_rootStatistics.number_of_visits + 1;
    pruneToLiveNodeLimit(std::max<uint32>(1, liveNodeLimit));
}

void SearchTree::prepareForSearch(const uint32 visitLimit, const uint32 parallelSearches) {
    if (parallelSearches == 0) {
        throw std::invalid_argument("SearchTree parallel search count must be positive");
    }

    uint64 maximumNewNodes = parallelSearches;
    if (m_rootStatistics.number_of_visits < visitLimit) {
        maximumNewNodes = static_cast<uint64>(visitLimit - m_rootStatistics.number_of_visits) +
                          parallelSearches - 1U;
    }
    if (maximumNewNodes + 1U >= capacity()) {
        throw std::logic_error("SearchTree capacity cannot reserve search and reroot slots");
    }

    const uint32 retainedNodeLimit = capacity() - static_cast<uint32>(maximumNewNodes) - 1U;
    pruneToLiveNodeLimit(std::max<uint32>(1, retainedNodeLimit));
}

int SearchTree::maxDepth() const {
    int maximumDepth = 1;
    std::vector<std::pair<NodeIndex, int>> pending = {{m_rootIndex, 1}};
    while (!pending.empty()) {
        const auto [currentIndex, depth] = pending.back();
        pending.pop_back();
        maximumDepth = std::max(maximumDepth, depth);
        for (const Child &descendant : node(currentIndex).children) {
            if (descendant.node_index != INVALID_NODE_INDEX) {
                pending.emplace_back(descendant.node_index, depth + 1);
            }
        }
    }
    return maximumDepth;
}

uint64 SearchTree::totalChildCount() const {
    uint64 count = 0;
    for (const NodeSlot &slot : m_slots) {
        if (slot.value.has_value()) {
            count += slot.value->children.size();
        }
    }
    return count;
}

std::vector<MCTSChild> SearchTree::rootChildren() const {
    const SearchNode &root = node(m_rootIndex);
    std::vector<MCTSChild> snapshots;
    snapshots.reserve(root.children.size());
    for (const Child &rootChild : root.children) {
        snapshots.push_back({toString(rootChild.move), encodeMove(rootChild.move, &root.board),
                             rootChild.policy, rootChild.number_of_visits, rootChild.result_sum,
                             rootChild.virtual_loss, rootChild.node_index != INVALID_NODE_INDEX});
    }
    return snapshots;
}

std::string SearchTree::rootRepr() const {
    const SearchNode &root = node(m_rootIndex);
    std::stringstream output;
    output << "MCTSRoot(" << root.board.fen() << ", Move: " << toString(m_rootMove)
           << ", Visits: " << m_rootStatistics.number_of_visits
           << ", Score: " << m_rootStatistics.result_sum
           << ", Virtual Loss: " << m_rootStatistics.virtual_loss
           << ", Live Nodes: " << m_liveNodeCount << "/" << capacity() << ")";
    return output.str();
}

std::string SearchTree::rootMove() const { return toString(m_rootMove); }

void SearchTree::validateRoot(const NodeIndex index) const {
    if (index != m_rootIndex) {
        throw std::logic_error("Stale MCTSRoot handle");
    }
    static_cast<void>(node(index));
}

MCTSRoot MCTSRoot::create(Board board, const uint32 arenaCapacity) {
    auto tree = std::make_shared<SearchTree>(std::move(board), arenaCapacity);
    return MCTSRoot(tree, tree->rootIndex());
}

MCTSRoot MCTSRoot::create(const std::string &fen, const uint32 arenaCapacity) {
    return create(Board(fen), arenaCapacity);
}

SearchTree &MCTSRoot::tree() {
    m_tree->validateRoot(m_rootIndex);
    return *m_tree;
}

const SearchTree &MCTSRoot::tree() const {
    m_tree->validateRoot(m_rootIndex);
    return *m_tree;
}

NodeIndex MCTSRoot::rootIndex() const {
    static_cast<void>(tree());
    return m_rootIndex;
}

const Board &MCTSRoot::board() const { return tree().node(m_rootIndex).board; }

bool MCTSRoot::isTerminal() const { return tree().node(m_rootIndex).isTerminal(); }

bool MCTSRoot::isExpanded() const { return tree().node(m_rootIndex).isExpanded(); }

uint32 MCTSRoot::visits() const { return tree().rootStatistics().number_of_visits; }

float MCTSRoot::virtualLoss() const { return tree().rootStatistics().virtual_loss; }

float MCTSRoot::resultSum() const { return tree().rootStatistics().result_sum; }

int MCTSRoot::maxDepth() const { return tree().maxDepth(); }

uint32 MCTSRoot::liveNodeCount() const { return tree().liveNodeCount(); }

uint64 MCTSRoot::totalChildCount() const { return tree().totalChildCount(); }

uint32 MCTSRoot::arenaCapacity() const { return tree().capacity(); }

std::vector<MCTSChild> MCTSRoot::children() const { return tree().rootChildren(); }

std::string MCTSRoot::move() const { return tree().rootMove(); }

std::string MCTSRoot::repr() const { return tree().rootRepr(); }

MCTSRoot MCTSRoot::makeNewRoot(const uint32 childIndex) {
    m_rootIndex = tree().reroot(childIndex);
    return MCTSRoot(m_tree, m_rootIndex);
}

void MCTSRoot::discount(const float percentageOfNodeVisitsToKeep) {
    tree().discount(percentageOfNodeVisitsToKeep);
}
