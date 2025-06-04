#include "MCTSNode.hpp"

// #include "../MoveEncoding.hpp"


MCTSNode MCTSNode::root(Board board) {
    MCTSNode instance(std::move(board), 1.0, Move::null(), nullptr);
    instance.number_of_visits = 1;
    return instance;
}

MCTSNode::MCTSNode(Board board, float policy, Move move_to_get_here, MCTSNode *parent)
    : board(std::move(board)), parent(parent), move_to_get_here(move_to_get_here),
       policy(policy) {}

float MCTSNode::ucb(float cParam) const {
    if (!parent) {
        throw std::logic_error("Node must have a parent");
    }

    float uScore = policy * cParam * std::sqrt(parent->number_of_visits) / (1 + number_of_visits);

    float qScore = 0.0;
    if (number_of_visits > 0) {
        qScore = -1 * (result_score + virtual_loss) / number_of_visits;
    } else {
        qScore = -1.0; // Default to loss for unvisited moves
    }

    return uScore + qScore;
}

MCTSNode::MCTSNode(MCTSNode &&other)
    : board(std::move(other.board)), parent(other.parent), children(std::move(other.children)),
      move_to_get_here(other.move_to_get_here), number_of_visits(other.number_of_visits),
      virtual_loss(other.virtual_loss), result_score(other.result_score), policy(other.policy) {
    // NOTE: For some reason this move constructor is required to set the parent pointer
    // correctly in some part of the MCTS search.
    for (MCTSNode &child : children) {
        child.parent = this;
    }
}

void MCTSNode::expand(const std::vector<MoveScore> &moves_with_scores) {
    if (isFullyExpanded() || moves_with_scores.empty())
        return;

    children.reserve(moves_with_scores.size());

    for (const auto &[move, score] : moves_with_scores) {
        Board newBoard = board.copy();
        try {
            newBoard.push(move);

            children.emplace_back(std::move(newBoard), score, move, this);
        } catch (std::invalid_argument &e) {
            // Ignore invalid moves (e.g., castling when the king is in check)
            log("Invalid move: ", e.what(), " for move: ", move, " in board: ", board.fen());
        }
    }
}

void MCTSNode::backPropagate(float result) {
    MCTSNode* node = this;

    while (node) {
        node->result_score += result;
        node->number_of_visits += 1;

        result = -result * 0.99; // Discount the result for the parent
        node = node->parent;
    }
}

void MCTSNode::updateVirtualLoss(int delta) {
    MCTSNode* node = this;

    // NOTE: more virtual loss, to avoid the same node being selected multiple times (i.e. muliply delta by 100?)
    delta *= 3;

    while (node) {
        node->virtual_loss += delta;
        node->number_of_visits += delta;

        node = node->parent;
    }
}

MCTSNode &MCTSNode::bestChild(float cParam) {
    assert(!children.empty() && "Node has no children");

    MCTSNode *bestChild = &children[0];
    float bestScore = children[0].ucb(cParam);

    for (size_t i = 1; i < children.size(); ++i) {
        float score = children[i].ucb(cParam);
        if (score > bestScore) {
            bestScore = score;
            bestChild = &children[i];
        }
    }

    return *bestChild;
}