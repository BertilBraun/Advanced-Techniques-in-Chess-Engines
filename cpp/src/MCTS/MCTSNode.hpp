#pragma once

#include "common.hpp"

#include "../MoveEncoding.hpp"

class MCTSNode {
public:
    static MCTSNode root(Board board);

    MCTSNode(Board board, float policy, int encoded_move_to_get_here, MCTSNode *parent,
             int num_played_moves);

    bool isTerminalNode() { return board.isGameOver(); }

    bool isFullyExpanded() const { return !children.empty(); }

    float ucb(float c_param = 0.1) const;

    void expand(const std::vector<MoveScore> &moves_with_scores);

    void backPropagate(float result);

    void updateVirtualLoss(int delta);

    MCTSNode &bestChild(float cParam);

    bool operator==(MCTSNode &other) {
        return board == other.board && encoded_move_to_get_here == other.encoded_move_to_get_here;
    }

    Board board;
    MCTSNode *parent;
    std::vector<MCTSNode> children;
    int encoded_move_to_get_here;
    int num_played_moves = 0;
    int number_of_visits = 0;
    float virtual_loss = 1.0; // Init to loss as that seems to work better during search
    float result_score = 0.0;
    float policy = 0.0;
};
