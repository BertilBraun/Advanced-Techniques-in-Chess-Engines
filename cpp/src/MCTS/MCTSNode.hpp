#pragma once

#include "common.hpp"

class NodePool;

class MCTSNode {
public:
    MCTSNode() {}

    MCTSNode(const std::string &boardFen, float policy, Move move_to_get_here, NodeId parent,
             NodePool *pool);

    bool isTerminalNode() const { return board.isGameOver(); }

    bool isFullyExpanded() const { return !children.empty(); }

    float ucb(float uCommon, float parentScore) const;

    void expand(const std::vector<MoveScore> &moves_with_scores);

    void backPropagate(float result);

    void updateVirtualLoss(int delta);

    NodeId bestChild(float cParam) const;

    bool operator==(const MCTSNode &other) const {
        return board.quickHash() == other.board.quickHash() &&
               move_to_get_here == other.move_to_get_here;
    }

    std::string repr() const {
        std::stringstream ss;
        ss << "MCTSNode(" << board.fen() << ", Move: " << toString(move_to_get_here)
           << ", Visits: " << number_of_visits << ", Score: " << result_score
           << ", Policy: " << policy << ")";
        return ss.str();
    }

    NodeId parent = INVALID_NODE;
    NodeId myId = INVALID_NODE;
    std::vector<NodeId> children;

    NodePool *pool = nullptr;

    Board board;
    Move move_to_get_here = Move::null();
    int number_of_visits = 0;
    float virtual_loss = 0.0;
    float result_score = 0.0;
    float policy = 0.0;
};
