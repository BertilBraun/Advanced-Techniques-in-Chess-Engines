#pragma once

#include "common.hpp"

class MCTSNode {
public:
    static MCTSNode root(Board board);

    MCTSNode(Board board, float policy, Move move_to_get_here, MCTSNode *parent);

    MCTSNode(MCTSNode &&other);

    bool isTerminalNode() { return board.isGameOver(); }

    bool isFullyExpanded() const { return !children.empty(); }

    float ucb(float cParam) const;

    void expand(const std::vector<MoveScore> &moves_with_scores);

    void backPropagate(float result);

    void updateVirtualLoss(int delta);

    MCTSNode &bestChild(float cParam);

    bool operator==(MCTSNode &other) {
        return board == other.board && move_to_get_here == other.move_to_get_here;
    }

    Board board;
    MCTSNode *parent;
    std::vector<MCTSNode> children;
    Move move_to_get_here;
    int number_of_visits = 0;
    float virtual_loss = 0.0;
    float result_score = 0.0;
    float policy = 0.0;
};
