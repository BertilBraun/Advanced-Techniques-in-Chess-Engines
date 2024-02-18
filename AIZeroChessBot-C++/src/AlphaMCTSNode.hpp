#pragma once

#include "common.hpp"

class AlphaMCTSNode {
public:
    static AlphaMCTSNode *root(Board board) {
        AlphaMCTSNode *instance =
            new AlphaMCTSNode(std::move(board), 1.0, Move::null(), nullptr, 0);
        instance->number_of_visits = 1.0;
        return instance;
    }

    AlphaMCTSNode(Board board, float policy, Move move_to_get_here, AlphaMCTSNode *parent,
                  int num_played_moves)
        : board(std::move(board)), parent(parent), move_to_get_here(move_to_get_here),
          num_played_moves(num_played_moves), number_of_visits(0.0001f), result_score(-1.0),
          policy(policy) {}

    bool isTerminalNode() { return board.isGameOver(); }

    bool isFullyExpanded() const { return !children.empty(); }

    float ucb(float c_param = 0.1) const {
        if (!parent) {
            throw std::logic_error("Node must have a parent");
        }

        float ucb_score =
            policy * c_param * std::sqrt(parent->number_of_visits) / (1 + number_of_visits);

        if (number_of_visits > 0) {
            ucb_score += 1 - ((result_score / number_of_visits) + 1) / 2;
        }
        return ucb_score;
    }

    void expand(const std::vector<std::pair<Move, float>> &moves_with_scores) {
        std::vector<float> policies;

        children.reserve(moves_with_scores.size());
        policies.reserve(moves_with_scores.size());

        for (const auto &[move, score] : moves_with_scores) {
            Board new_board = board.copy();
            new_board.push(move);

            children.emplace_back(std::move(new_board), score, move, this, num_played_moves + 1);
            policies.push_back(score);
        }

        // Initialize tensors for vectorized operations
        children_number_of_visits = torch::zeros({static_cast<long>(moves_with_scores.size())});
        children_result_scores = torch::zeros({static_cast<long>(moves_with_scores.size())});
        children_policies = torch::from_blob(policies.data(), {static_cast<long>(policies.size())},
                                             torch::kFloat32);
    }

    void backPropagate(float result) {
        number_of_visits += 1.0;
        result_score += result;
        if (parent) {
            // Vectorized update of visits and scores
            auto child_index =
                std::distance(parent->children.begin(),
                              std::find_if(parent->children.begin(), parent->children.end(),
                                           [this](const auto &child) { return &child == this; }));
            parent->children_number_of_visits.index_put_(
                {child_index}, parent->children_number_of_visits.index({child_index}) + 1.0);
            parent->children_result_scores.index_put_(
                {child_index}, parent->children_result_scores.index({child_index}) + result);
            parent->backPropagate(result);
        }
    }

    AlphaMCTSNode &bestChild(float c_param = 0.1) {
        assert(!children.empty() && "Node has no children");

        auto q_scores = 1 - ((children_result_scores / children_number_of_visits) + 1) / 2;
        auto policy_scores = children_policies * c_param *
                             torch::sqrt(torch::tensor(number_of_visits)) /
                             (1 + children_number_of_visits);
        auto ucb_scores = q_scores + policy_scores;

        auto best_child_index = ucb_scores.argmax(0).item<int64_t>();

        return children[best_child_index];
    }

    bool operator==(AlphaMCTSNode &other) {
        return board == other.board && move_to_get_here == other.move_to_get_here;
    }

    Board board;
    AlphaMCTSNode *parent;
    std::vector<AlphaMCTSNode> children;
    Move move_to_get_here;
    int num_played_moves;
    float number_of_visits;
    float result_score;
    float policy;

private:
    torch::Tensor children_number_of_visits;
    torch::Tensor children_result_scores;
    torch::Tensor children_policies;
};
