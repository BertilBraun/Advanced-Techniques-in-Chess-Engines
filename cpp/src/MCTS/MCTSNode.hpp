#pragma once

#include "common.hpp"

class NodePool;

class MCTSNode {
public:
    static NodeId root(const std::string &boardFen, NodePool *pool);

    MCTSNode(const std::string &boardFen, float policy, Move move_to_get_here, NodeId parent,
             NodePool *pool);

    bool isTerminalNode() const { return board.is_game_over(); }

    bool isFullyExpanded() const { return !children.empty(); }

    float ucb(float uCommon) const;

    void expand(const std::vector<MoveScore> &moves_with_scores);

    void backPropagate(float result) const;

    void updateVirtualLoss(int delta) const;

    NodeId bestChild(float cParam) const;

    bool operator==(const MCTSNode &other) const {
        return board.quick_hash() == other.board.quick_hash() &&
               move_to_get_here == other.move_to_get_here;
    }

    NodeId parent = INVALID_NODE;
    NodeId myId;
    std::vector<NodeId> children;

    NodePool *pool;

    Board board;
    Move move_to_get_here;
    int number_of_visits = 0;
    float virtual_loss = 0.0;
    float result_score = 0.0;
    float policy = 0.0;
};
