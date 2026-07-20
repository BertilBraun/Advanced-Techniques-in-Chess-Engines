#pragma once

#include "common.hpp"

using NodeIndex = uint64;

inline constexpr NodeIndex INVALID_NODE_INDEX = std::numeric_limits<NodeIndex>::max();

struct Child {
    Move move;
    float policy;
    uint32 number_of_visits = 0;
    float result_sum = 0.0f;
    float virtual_loss = 0.0f;
    NodeIndex node_index = INVALID_NODE_INDEX;

    Child(Move move, float policy) : move(move), policy(policy) {}
};

struct SearchNode {
    Board board;
    std::vector<Child> children;
    NodeIndex parent_index;
    uint32 parent_child_index;

    SearchNode(Board board, NodeIndex parent_index, uint32 parent_child_index)
        : board(std::move(board)), parent_index(parent_index),
          parent_child_index(parent_child_index) {}

    [[nodiscard]] bool isExpanded() const { return !children.empty(); }
    [[nodiscard]] bool isTerminal() const { return board.isGameOver(); }
};

struct RootStatistics {
    uint32 number_of_visits = 0;
    float result_sum = 0.0f;
    float virtual_loss = 0.0f;
};

struct MCTSChild {
    std::string move;
    int encoded_move;
    float policy;
    uint32 visits;
    float result_sum;
    float virtual_loss;
    bool is_materialized;
};

class SearchTree {
public:
    SearchTree(Board rootBoard, uint32 capacity);

    SearchTree(const SearchTree &) = delete;
    SearchTree &operator=(const SearchTree &) = delete;
    SearchTree(SearchTree &&) = delete;
    SearchTree &operator=(SearchTree &&) = delete;

    [[nodiscard]] uint32 capacity() const { return static_cast<uint32>(m_slots.size()); }
    [[nodiscard]] uint32 liveNodeCount() const { return m_liveNodeCount; }
    [[nodiscard]] uint64 totalChildCount() const;
    [[nodiscard]] NodeIndex rootIndex() const { return m_rootIndex; }
    [[nodiscard]] const RootStatistics &rootStatistics() const { return m_rootStatistics; }

    [[nodiscard]] SearchNode &node(NodeIndex index);
    [[nodiscard]] const SearchNode &node(NodeIndex index) const;
    [[nodiscard]] Child &child(NodeIndex parentIndex, uint32 childIndex);
    [[nodiscard]] const Child &child(NodeIndex parentIndex, uint32 childIndex) const;

    void expand(NodeIndex nodeIndex, const std::vector<MoveScore> &movesWithScores);
    [[nodiscard]] NodeIndex materializeChild(NodeIndex parentIndex, uint32 childIndex);
    [[nodiscard]] uint32 bestChildIndex(NodeIndex parentIndex, float cParam) const;

    void addVirtualLoss(NodeIndex leafIndex);
    void backPropagate(NodeIndex leafIndex, float result);
    void backPropagateAndRemoveVirtualLoss(NodeIndex leafIndex, float result);

    [[nodiscard]] NodeIndex reroot(uint32 childIndex);
    void discount(float percentageOfNodeVisitsToKeep);
    void prepareForSearch(uint32 visitLimit, uint32 parallelSearches);

    [[nodiscard]] int maxDepth() const;
    [[nodiscard]] std::vector<MCTSChild> rootChildren() const;
    [[nodiscard]] std::string rootMove() const;
    [[nodiscard]] std::string rootRepr() const;

private:
    struct NodeSlot {
        uint32 generation = 0;
        std::optional<SearchNode> value;
    };

    std::vector<NodeSlot> m_slots;
    std::vector<uint32> m_freeSlots;
    uint32 m_liveNodeCount = 0;
    NodeIndex m_rootIndex = INVALID_NODE_INDEX;
    RootStatistics m_rootStatistics;
    Move m_rootMove = Move::null();

    [[nodiscard]] static NodeIndex makeIndex(uint32 slotIndex, uint32 generation);
    [[nodiscard]] static uint32 slotIndex(NodeIndex index);
    [[nodiscard]] static uint32 generation(NodeIndex index);
    [[nodiscard]] NodeIndex allocateNode(Board board, NodeIndex parentIndex,
                                         uint32 parentChildIndex);
    void releaseNode(NodeIndex index);
    void reclaimSubtree(NodeIndex index);
    void pruneToLiveNodeLimit(uint32 liveNodeLimit);
    void validateRoot(NodeIndex index) const;
    [[nodiscard]] RootStatistics nodeStatistics(NodeIndex index) const;
    static void discountStatistics(uint32 &numberOfVisits, float &resultSum,
                                   float percentageOfNodeVisitsToKeep);

    friend class MCTSRoot;
};

class MCTSRoot {
public:
    static MCTSRoot create(Board board, uint32 arenaCapacity);
    static MCTSRoot create(const std::string &fen, uint32 arenaCapacity);

    [[nodiscard]] const Board &board() const;
    [[nodiscard]] bool isTerminal() const;
    [[nodiscard]] bool isExpanded() const;
    [[nodiscard]] uint32 visits() const;
    [[nodiscard]] float virtualLoss() const;
    [[nodiscard]] float resultSum() const;
    [[nodiscard]] int maxDepth() const;
    [[nodiscard]] uint32 liveNodeCount() const;
    [[nodiscard]] uint64 totalChildCount() const;
    [[nodiscard]] uint32 arenaCapacity() const;
    [[nodiscard]] std::vector<MCTSChild> children() const;
    [[nodiscard]] std::string move() const;
    [[nodiscard]] std::string repr() const;

    [[nodiscard]] MCTSRoot makeNewRoot(uint32 childIndex);
    void discount(float percentageOfNodeVisitsToKeep);

    [[nodiscard]] SearchTree &tree();
    [[nodiscard]] const SearchTree &tree() const;
    [[nodiscard]] NodeIndex rootIndex() const;

private:
    MCTSRoot(std::shared_ptr<SearchTree> tree, NodeIndex rootIndex)
        : m_tree(std::move(tree)), m_rootIndex(rootIndex) {}

    std::shared_ptr<SearchTree> m_tree;
    NodeIndex m_rootIndex;
};
