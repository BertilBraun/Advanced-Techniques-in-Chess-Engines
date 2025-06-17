#include "MCTSNode.hpp"

#include "NodePool.h"

// NOTE: more virtual loss, to avoid the same node being selected multiple times?
// (i.e. multiply delta by 2-5?)
constexpr int VIRTUAL_LOSS_DELTA = 1; // How much to increase the virtual loss by each time

constexpr float TURN_DISCOUNT = 0.99f; // Discount factor for the result score when backpropagating
// this makes the result score decay over time, simulating the fact that very long searches add
// more uncertainty to the result.

constexpr float FPU_REDUCTION = 0.10f;      // ≈-10 centipawns


MCTSNode::MCTSNode(const std::string &boardFen, const float policy, const Move move_to_get_here,
                   const NodeId parent, NodePool *pool)
    : parent(parent), pool(pool), board(boardFen), move_to_get_here(move_to_get_here),
      policy(policy) {}

float MCTSNode::ucb(const float uCommon, const float parentScore) const {
    TIMEIT("MCTSNode::ucb");
    const float uScore = policy * uCommon / static_cast<float>(1 + number_of_visits);

    // TODO which is the best initializer for qScore?
    // most seem to init to 0.0
    // CrazyAra inits to -1.0
    // LeelaZero inits to parentScore
    float qScore = 0.0; // parentScore - FPU_REDUCTION;
    if (number_of_visits > 0) {
        qScore = -1 * (result_score + virtual_loss) / static_cast<float>(number_of_visits);
    }

    return uScore + qScore;
}

void MCTSNode::expand(const std::vector<MoveScore> &moves_with_scores) {
    TIMEIT("MCTSNode::expand");

    if (isFullyExpanded() || moves_with_scores.empty())
        return;

    children.reserve(moves_with_scores.size());

    for (const auto &[move, score] : moves_with_scores) {
        Board moveBoard = board; // Create a copy of the board to make the move
        moveBoard.makeMove(move);

        MCTSNode *child = pool->allocateNode(moveBoard.fen(), score, move, myId, pool);

        children.emplace_back(child->myId);
    }
}

void MCTSNode::backPropagate(float result) {
    TIMEIT("MCTSNode::backPropagate");

    MCTSNode *node = this;

    while (node) {
        node->result_score += result; // Add the result to the node's score
        node->number_of_visits += 1;  // Increment the visit count

        result = -1.0f * result * TURN_DISCOUNT; // Discount the result for the parent
        node = (node->parent == INVALID_NODE) ? nullptr : pool->get(node->parent);
    }
}

void MCTSNode::backPropagateAndRemoveVirtualLoss(float result) {
    TIMEIT("MCTSNode::backPropagateAndRemoveVirtualLoss");

    MCTSNode *node = this;

    while (node) {
        node->result_score += result;             // Add the result to the node's score
        node->virtual_loss -= VIRTUAL_LOSS_DELTA; // Remove the virtual loss
        // NOTE: Do not change the visit count here, as that is already done in addVirtualLoss

        result = -1.0f * result * TURN_DISCOUNT; // Discount the result for the parent
        node = (node->parent == INVALID_NODE) ? nullptr : pool->get(node->parent);
    }
}

void MCTSNode::addVirtualLoss() {
    TIMEIT("MCTSNode::updateVirtualLoss");

    MCTSNode *node = this;

    while (node) {
        node->virtual_loss += VIRTUAL_LOSS_DELTA; // Update the virtual loss
        node->number_of_visits += 1;              // Increment the visit count

        node = (node->parent == INVALID_NODE) ? nullptr : pool->get(node->parent);
    }
}

NodeId MCTSNode::bestChild(const float cParam) const {
    TIMEIT("MCTSNode::bestChild");

    assert(!children.empty() && "Node has no children");

    const float uCommon = cParam * std::sqrt(number_of_visits);
    const float parentScore = result_score / static_cast<float>(number_of_visits);

    NodeId bestChildId = children[0];
    float bestScore = pool->get(children[0])->ucb(uCommon, parentScore);

    for (size_t i = 1; i < children.size(); ++i) {
        const float score = pool->get(children[i])->ucb(uCommon, parentScore);
        if (score > bestScore) {
            bestScore = score;
            bestChildId = children[i];
        }
    }

    return bestChildId;
}