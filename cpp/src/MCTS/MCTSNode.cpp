#include "MCTSNode.hpp"

#include "NodePool.h"

NodeId MCTSNode::root(const std::string &boardFen, NodePool *pool) {
    const NodeId instanceId = pool->allocateNode(boardFen, 1.0, Move::null(), INVALID_NODE, pool);
    MCTSNode &instance = pool->get(instanceId);
    instance.myId = instanceId;
    instance.number_of_visits = 1; // Initialize visits to 1 for the root node
    return instanceId;
}

MCTSNode::MCTSNode(const std::string &boardFen, const float policy, const Move move_to_get_here,
                   const NodeId parent, NodePool *pool)
    : parent(parent), myId(INVALID_NODE), pool(pool), board(boardFen),
      move_to_get_here(move_to_get_here), policy(policy) {}

float MCTSNode::ucb(const float uCommon) const {
    const float uScore = policy * uCommon / (1 + number_of_visits);

    float qScore = -1.0; // Default to loss for unvisited moves
    // TODO most seem to init to 0.0
    // CrazyAra inits to -1
    if (number_of_visits > 0) {
        qScore = -1 * (result_score + virtual_loss) / number_of_visits;
    }

    return uScore + qScore;
}

void MCTSNode::expand(const std::vector<MoveScore> &moves_with_scores) {
    if (isFullyExpanded() || moves_with_scores.empty())
        return;

    children.reserve(moves_with_scores.size());

    for (const auto &[move, score] : moves_with_scores) {
        Board moveBoard(board.fen());
        moveBoard.make_move(move);

        NodeId childId = pool->allocateNode(moveBoard.fen(), score, move, myId, pool);
        pool->get(childId).myId = childId;

        children.emplace_back(childId);
    }
}

void MCTSNode::backPropagate(float result) const {
    NodeId nodeId = myId;

    while (nodeId != INVALID_NODE) {
        MCTSNode &node = pool->get(nodeId);
        node.result_score += result; // Add the result to the node's score
        node.number_of_visits += 1;  // Increment the visit count

        result = -result * 0.99; // Discount the result for the parent
        nodeId = node.parent;
    }
}

void MCTSNode::updateVirtualLoss(int delta) const {
    NodeId nodeId = myId;

    // NOTE: more virtual loss, to avoid the same node being selected multiple times (i.e. multiply
    // delta by 100?)
    delta *= 3;

    while (nodeId != INVALID_NODE) {
        MCTSNode &node = pool->get(nodeId);
        node.virtual_loss += delta;
        node.number_of_visits += delta;

        nodeId = node.parent;
    }
}

NodeId MCTSNode::bestChild(const float cParam) const {
    assert(!children.empty() && "Node has no children");

    const float uCommon = cParam * std::sqrt(number_of_visits);

    NodeId bestChildId = children[0];
    float bestScore = pool->get(children[0]).ucb(uCommon);

    for (size_t i = 1; i < children.size(); ++i) {
        const float score = pool->get(children[i]).ucb(uCommon);
        if (score > bestScore) {
            bestScore = score;
            bestChildId = children[i];
        }
    }

    return bestChildId;
}