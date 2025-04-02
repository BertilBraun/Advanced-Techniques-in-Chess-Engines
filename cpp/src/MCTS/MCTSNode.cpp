#include "MCTSNode.hpp"
MCTSNode MCTSNode::root(Board board) {
    MCTSNode instance(std::move(board), 1.0, -1, nullptr, 0);
    instance.number_of_visits = 1;
    return instance;
}
MCTSNode::MCTSNode(Board board, float policy, int encoded_move_to_get_here, MCTSNode *parent,
                   int num_played_moves)
    : board(std::move(board)), parent(parent), encoded_move_to_get_here(encoded_move_to_get_here),
      num_played_moves(num_played_moves), policy(policy) {}
float MCTSNode::ucb(float c_param) const {
    if (!parent) {
        throw std::logic_error("Node must have a parent");
    }

    float ucb_score =
        policy * c_param * std::sqrt(parent->number_of_visits) / (1 + number_of_visits);

    if (number_of_visits > 0) {
        ucb_score += 1 - (((result_score + virtual_loss) / number_of_visits) + 1) / 2;
    }
    return ucb_score;
}
void MCTSNode::expand(const std::vector<MoveScore> &moves_with_scores) {
    children.reserve(moves_with_scores.size());

    for (const auto &[move, score] : moves_with_scores) {
        Board new_board = board.copy();
        new_board.push(decodeMove(move)); // TODO check whether this is a bottleneck

        children.emplace_back(std::move(new_board), score, move, this, num_played_moves + 1);
    }
}
void MCTSNode::backPropagate(float result) {
    result_score += result;
    number_of_visits += 1;
    if (parent) {
        parent->backPropagate(-result * 0.99); // Discount the result for the parent
    }
}
void MCTSNode::updateVirtualLoss(int delta) {
    virtual_loss += delta;
    number_of_visits += delta;
    if (parent) {
        parent->updateVirtualLoss(delta);
    }
}
MCTSNode &MCTSNode::bestChild(float cParam) {
    assert(!children.empty() && "Node has no children");

    MCTSNode &bestChild = children[0];
    float bestScore = children[0].ucb(cParam);

    for (size_t i = 1; i < children.size(); ++i) {
        float score = children[i].ucb(cParam);
        if (score > bestScore) {
            bestScore = score;
            bestChild = children[i];
        }
    }

    return bestChild;
}