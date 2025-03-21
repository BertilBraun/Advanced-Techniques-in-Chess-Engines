#pragma once

#include "common.hpp"

class AlphaMCTSNode {
public:
    static AlphaMCTSNode root(Board board) {
        AlphaMCTSNode instance(std::move(board), 1.0, Move::null(), nullptr, 0);
        instance.number_of_visits = 1;
        return instance;
    }

    AlphaMCTSNode(Board board, float policy, Move move_to_get_here, AlphaMCTSNode *parent,
                  int num_played_moves)
        : board(std::move(board)), parent(parent), move_to_get_here(move_to_get_here),
          num_played_moves(num_played_moves), policy(policy) {}

    bool isTerminalNode() { return this->board.isGameOver(); }

    bool isFullyExpanded() const { return !this->children.empty(); }

    float ucb(float c_param = 0.1) const {
        if (!parent) {
            throw std::logic_error("Node must have a parent");
        }

        float ucb_score = this->policy * c_param * std::sqrt(this->parent->number_of_visits) /
                          (1 + this->number_of_visits);

        if (this->number_of_visits > 0) {
            ucb_score +=
                1 - (((this->result_score + this->virtual_loss) / this->number_of_visits) + 1) / 2;
        }
        return ucb_score;
    }

    void expand(const std::vector<std::pair<Move, float>> &moves_with_scores) {
        this->children.reserve(moves_with_scores.size());

        for (const auto &[move, score] : moves_with_scores) {
            Board new_board = this->board.copy();
            new_board.push(move); // TODO check whether this is a bottleneck

            this->children.emplace_back(std::move(new_board), score, move, this,
                                        num_played_moves + 1);
        }
    }

    void backPropagate(float result) {
        this->result_score += result;
        this->number_of_visits += 1;
        if (this->parent) {
            this->parent->backPropagate(-result * 0.99); // Discount the result for the parent
        }
    }

    void updateVirtualLoss(int delta) {
        this->virtual_loss += delta;
        this->number_of_visits += delta;
        if (this->parent) {
            this->parent->updateVirtualLoss(delta);
        }
    }

    AlphaMCTSNode &bestChild(float cParam) {
        assert(!this->children.empty() && "Node has no children");

        AlphaMCTSNode &bestChild = this->children[0];
        float bestScore = this->children[0].ucb(cParam);

        for (size_t i = 1; i < this->children.size(); ++i) {
            float score = this->children[i].ucb(cParam);
            if (score > bestScore) {
                bestScore = score;
                bestChild = this->children[i];
            }
        }

        return bestChild;
    }

    bool operator==(AlphaMCTSNode &other) {
        return board == other.board && move_to_get_here == other.move_to_get_here;
    }

    Board board;
    AlphaMCTSNode *parent;
    std::vector<AlphaMCTSNode> children;
    Move move_to_get_here;
    int num_played_moves = 0;
    int number_of_visits = 0;
    float virtual_loss = 1.0; // Init to loss as that seems to work better during search
    float result_score = 0.0;
    float policy = 0.0;
};
