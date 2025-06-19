#include "MCTSNode.hpp"

// NOTE: more virtual loss, to avoid the same node being selected multiple times?
// (i.e. multiply delta by 2-5?)
constexpr int VIRTUAL_LOSS_DELTA = 1; // How much to increase the virtual loss by each time

constexpr float TURN_DISCOUNT = 0.99f; // Discount factor for the result score when backpropagating
// this makes the result score decay over time, simulating the fact that very long searches add
// more uncertainty to the result.

constexpr float FPU_REDUCTION = 0.10f; // ≈-10 centipawns

std::shared_ptr<MCTSNode> MCTSNode::createRoot(const std::string &fen) {
    return std::shared_ptr<MCTSNode>(
        new MCTSNode(fen, 1.0f, Move::null(), std::weak_ptr<MCTSNode>()));
}

MCTSNode::MCTSNode(const std::string &fen, const float policy, const Move move,
                   std::weak_ptr<MCTSNode> parent)
    : parent(std::move(parent)), board(fen), move_to_get_here(move), policy(policy) {}

float MCTSNode::ucb(const float uCommon, const float parentQ) const {
    TIMEIT("MCTSNode::ucb");
    const float uScore = policy * uCommon / static_cast<float>(1 + number_of_visits);

    // TODO which is the best initializer for qScore?
    // most seem to init to 0.0
    // CrazyAra inits to -1.0
    // LeelaZero inits to parentQ
    float qScore = 0.0f; //  parentQ - FPU_REDUCTION;
    if (number_of_visits > 0) {
        qScore = -1 * (result_sum + virtual_loss) / static_cast<float>(number_of_visits);
    }

    return uScore + qScore;
}

void MCTSNode::expand(const std::vector<MoveScore> &moves_with_scores) {
    TIMEIT("MCTSNode::expand");

    if (isExpanded() || moves_with_scores.empty())
        return;

    children.reserve(moves_with_scores.size());

    for (const auto &[move, score] : moves_with_scores) {
        Board next = board; // Create a copy of the board to make the move
        next.makeMove(move);

        children.emplace_back(
            std::shared_ptr<MCTSNode>(new MCTSNode(next.fen(), score, move, weak_from_this())));
    }
}

void MCTSNode::backPropagate(float result) {
    TIMEIT("MCTSNode::backPropagate");

    /* start with *this* kept alive by a shared ptr */
    std::shared_ptr<MCTSNode> node = shared_from_this();

    while (node) {
        node->result_sum += result;  // Add the result to the node's score
        node->number_of_visits += 1; // Increment the visit count

        result = -1.0f * result * TURN_DISCOUNT; // Discount the result for the parent
        node = node->parent.lock();
    }
}

void MCTSNode::backPropagateAndRemoveVirtualLoss(float result) {
    TIMEIT("MCTSNode::backPropagateAndRemoveVirtualLoss");

    /* start with *this* kept alive by a shared ptr */
    std::shared_ptr<MCTSNode> node = shared_from_this();

    while (node) {
        node->result_sum += result;               // Add the result to the node's score
        node->virtual_loss -= VIRTUAL_LOSS_DELTA; // Remove the virtual loss
        // NOTE: Do not change the visit count here, as that is already done in addVirtualLoss

        result = -1.0f * result * TURN_DISCOUNT; // Discount the result for the parent
        node = node->parent.lock();
    }
}

void MCTSNode::addVirtualLoss() {
    TIMEIT("MCTSNode::addVirtualLoss");

    /* start with *this* kept alive by a shared ptr */
    std::shared_ptr<MCTSNode> node = shared_from_this();

    while (node) {
        node->virtual_loss += VIRTUAL_LOSS_DELTA; // Update the virtual loss
        node->number_of_visits += 1;              // Increment the visit count

        node = node->parent.lock();
    }
}

std::shared_ptr<MCTSNode> MCTSNode::bestChild(const float cParam) const {
    TIMEIT("MCTSNode::bestChild");

    assert(!children.empty() && "Node has no children");

    const float uCommon = cParam * std::sqrt(number_of_visits);
    const float parentQ = result_sum / static_cast<float>(number_of_visits);

    auto best = children.front();
    float bestV = -std::numeric_limits<float>::infinity();

    for (const auto child : children) {
        const float v = child->ucb(uCommon, parentQ);
        if (v > bestV) {
            bestV = v;
            best = child;
        }
    }
    return best;
}

bool MCTSNode::operator==(const MCTSNode &other) const {
    return board.quickHash() == other.board.quickHash() &&
           move_to_get_here == other.move_to_get_here;
}

void MCTSNode::discount(const float percentage_of_node_visits_to_keep) {
    // Discount the node's score and visits.

    // Discount the node's score and visits. - Problem same divisor is not given because of
    // integer rounding
    number_of_visits =
        static_cast<int>(static_cast<float>(number_of_visits) * percentage_of_node_visits_to_keep);
    result_sum = static_cast<float>(static_cast<int>(
        static_cast<float>(static_cast<int>(result_sum)) * percentage_of_node_visits_to_keep));

    result_sum = clamp(result_sum, static_cast<float>(-number_of_visits),
                       static_cast<float>(number_of_visits));

    for (const auto &child : children)
        child->discount(percentage_of_node_visits_to_keep);
}

std::shared_ptr<MCTSNode> MCTSNode::makeNewRoot(const std::size_t childIdx) {
    assert(childIdx < children.size());
    auto newRoot = children[childIdx];
    newRoot->parent.reset(); // sever upward link
    children.clear(); // drop every other subtree from myself, so that the new root is the only
                      // child remaining
    return newRoot;
}

std::string MCTSNode::repr() const {
    std::stringstream ss;
    ss << "MCTSNode(" << board.fen() << ", Move: " << toString(move_to_get_here)
       << ", Visits: " << number_of_visits << ", Score: " << result_sum << ", Policy: " << policy
       << ", Virtual Loss: " << virtual_loss << ")";
    return ss.str();
}

int MCTSNode::maxDepth() const {
    int depth = 0;
    for (const auto &child : children) {
        const int childDepth = child->maxDepth();
        if (childDepth > depth)
            depth = childDepth;
    }
    return depth + 1; // +1 for the current node
}