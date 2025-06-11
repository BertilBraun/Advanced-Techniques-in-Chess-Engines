#include "MCTS.hpp"

#include "../BoardEncoding.hpp"
#include "MoveEncoding.hpp"

thread_local std::mt19937 randomEngine(std::random_device{}());

std::vector<float> dirichlet(const float alpha, const size_t n) {
    // Sample from a Dirichlet distribution with parameter alpha.
    std::gamma_distribution<float> gamma(alpha, 1.0);

    std::vector<float> noise(n);
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        noise[i] = gamma(randomEngine);
        sum += noise[i];
    }

    const float norm = 1.0f / sum;
    for (size_t i = 0; i < n; i++) {
        noise[i] *= norm;
    }

    return noise;
}

MCTSStatistics mctsStatistics(const MCTSNode *root, NodePool *pool) {
    MCTSStatistics stats;

    {
        // Compute the depth of each search tree using a DFS.
        std::function<int(const MCTSNode *)> dfs = [&](const MCTSNode *node) -> int {
            if (!node->isFullyExpanded())
                return 0;
            int maxDepth = 0;
            for (const NodeId child : node->children) {
                const int d = dfs(pool->get(child));
                if (d > maxDepth)
                    maxDepth = d;
            }
            return 1 + maxDepth;
        };
        stats.averageDepth = static_cast<float>(dfs(root));
    }
    {
        // Compute the entropy of the visit counts.
        auto entropy = [&](const MCTSNode *node) -> float {
            const int total = node->number_of_visits;
            float ent = 0.0f;
            for (const NodeId childId : node->children) {
                const MCTSNode *child = pool->get(childId);
                if (child->number_of_visits > 0) {
                    const float p = static_cast<float>(child->number_of_visits) / total;
                    ent -= p * std::log2(p);
                }
            }
            return ent;
        };

        stats.averageEntropy = entropy(root);
    }
    {
        auto klDivergence = [&](const MCTSNode *node) -> float {
            const int total = node->number_of_visits;
            float kl = 0.0f;
            for (const NodeId childId : node->children) {
                const MCTSNode *child = pool->get(childId);
                if (child->number_of_visits > 0) {
                    const float p = static_cast<float>(child->number_of_visits) / total;
                    const float q = 1.0f / node->children.size(); // Uniform distribution
                    if (p > 0.0f)
                        kl += p * std::log2(p / q);
                }
            }
            return kl;
        };

        stats.averageKLDivergence = klDivergence(root);
    }

    return stats;
}

MCTSResults MCTS::search(const std::vector<std::tuple<std::string, NodeId, int>> &boards) {
    if (boards.empty())
        return {.results = {}, .mctsStats = {}};

    std::vector<MCTSNode *> roots;
    roots.reserve(boards.size());

    std::vector<const Board *> newBoards;
    newBoards.reserve(boards.size());
    for (const auto &[fen, id, _] : boards) {
        if (id == INVALID_NODE) {
            MCTSNode *root = m_pool.allocateNode(fen, 1.0, Move::null(), INVALID_NODE, &m_pool);

            roots.push_back(root);

            newBoards.emplace_back(&root->board);
        } else {
            // If the node is already expanded, we can use it directly.
            // It will have to be cleaned up later. (i.e. its parent and siblings will be removed
            // and visits discounted)
            roots.push_back(m_pool.get(id));
        }
    }

    // Get policy moves (with noise) for the given boards.
    const std::vector<std::vector<MoveScore>> movesList = getPolicyWithNoise(newBoards);

    size_t moveIndex = 0;

    for (const auto root : roots) {
        if (root->parent == INVALID_NODE) {
            // If the node is not pre-expanded, we need to expand it with the moves.
            root->expand(movesList[moveIndex]);

            moveIndex++;
        }
    }

    const std::size_t N = boards.size();
    std::vector<std::future<std::tuple<MCTSResult, MCTSStatistics>>> futures;
    futures.reserve(N);

    for (const auto &[root, tup] : zip(roots, boards))
        futures.emplace_back(m_threadPool.enqueue(&MCTS::searchOneGame, this, root, get<2>(tup)));

    std::vector<MCTSResult> results;
    MCTSStatistics stats;
    results.reserve(N);
    for (auto &fut : futures) {
        const auto [result, rootStats] = fut.get();
        stats.averageDepth += rootStats.averageDepth;
        stats.averageEntropy += rootStats.averageEntropy;
        stats.averageKLDivergence += rootStats.averageKLDivergence;

        results.emplace_back(result);
    }

    stats.averageDepth /= static_cast<float>(N);
    stats.averageEntropy /= static_cast<float>(N);
    stats.averageKLDivergence /= static_cast<float>(N);

    return {.results = results, .mctsStats = stats};
}

void MCTS::parallelIterate(MCTSNode *root) {
    TimeItGuard timer("MCTS::parallelIterate");

    // These variables are initialized only once per thread
    // and retain their values between function calls
    thread_local std::vector<MCTSNode *> nodes;
    thread_local std::vector<const Board *> boards;

    // Clear contents but maintain capacity
    nodes.clear();
    boards.clear();
    nodes.reserve(m_args.num_parallel_searches);
    boards.reserve(m_args.num_parallel_searches);

    for (int _ : range(m_args.num_parallel_searches)) {
        MCTSNode * node = getBestChildOrBackPropagate(root, m_args.c_param);
        if (node != nullptr) {
            node->updateVirtualLoss(1);
            nodes.push_back(node);
            boards.emplace_back(&node->board);
        }
    }

    if (nodes.empty())
        return;

    // Run inference in batch.
    const std::vector<InferenceResult> results = m_client.inferenceBatch(boards);

    for (auto [node, result] : zip(nodes, results)) {
        const auto &[moves, value] = result;
        node->expand(moves);
        node->backPropagate(value);
        node->updateVirtualLoss(-1);
    }
}

std::vector<std::vector<MoveScore>>
MCTS::getPolicyWithNoise(const std::vector<const Board *> &boards) {
    const std::vector<InferenceResult> inferenceResults = m_client.inferenceBatch(boards);

    std::vector<std::vector<MoveScore>> noisyMoves;
    noisyMoves.reserve(inferenceResults.size());
    for (const auto &policy : inferenceResults | std::views::keys) {
        noisyMoves.push_back(addNoise(policy));
    }
    return noisyMoves;
}

std::vector<MoveScore> MCTS::addNoise(const std::vector<MoveScore> &moves) const {
    if (moves.empty())
        return {};

    const std::vector<float> noiseList = dirichlet(m_args.dirichlet_alpha, moves.size());

    std::vector<MoveScore> noisyMoves;
    noisyMoves.reserve(moves.size());
    for (const auto &[noise, moveScore] : zip(noiseList, moves)) {
        const auto &[move, score] = moveScore;
        float newScore = lerp(score, noise, m_args.dirichlet_epsilon);
        noisyMoves.emplace_back(move, newScore);
    }

    return noisyMoves;
}

// Traverse the tree to find the best child or, if the node is terminal,
// back-propagate the board’s result.
MCTSNode * MCTS::getBestChildOrBackPropagate(MCTSNode *root, const float cParam) {

    for (const NodeId childId : root->children) {
        MCTSNode *child = m_pool.get(childId);
        if (child->number_of_visits < m_args.min_visit_count) {
            // If the child has not been visited enough, we should traverse it.
            root = child;
            break;
        }
    }

    // We need to traverse the tree until we find a node that is not fully expanded
    MCTSNode *node = root;
    while (node->isFullyExpanded()) {
        node = m_pool.get(node->bestChild(cParam));
    }

    if (node->isTerminalNode()) {
        node->backPropagate(getBoardResultScore(node->board));
        return nullptr;
    }
    return node;
}

std::tuple<MCTSResult, MCTSStatistics> MCTS::searchOneGame(MCTSNode *root, int number_of_searches) {
    if (root->parent != INVALID_NODE) {
        // If the node has a parent, we need to clean it up first.
        std::function<void(NodeId, NodeId)> free = [&](const NodeId node, const NodeId excluded) {
            // Free the node and all its children, except the excluded one.
            if (node == excluded)
                return;

            for (const NodeId childId : m_pool.get(node)->children)
                free(childId, excluded);

            m_pool.deallocateNode(node);
        };

        // Remove the nodes parent and all its children from the pool.
        free(root->parent, root->myId);
        root->parent = INVALID_NODE;

        // Add Dirichlet noise to the node's policy.
        const std::vector<float> noise = dirichlet(m_args.dirichlet_alpha, root->children.size());

        for (const auto [i, childId] : enumerate(root->children)) {
            MCTSNode *child = m_pool.get(childId);
            child->policy = lerp(child->policy, noise[i], m_args.dirichlet_epsilon);
        }

        std::function<void(MCTSNode *)> discount = [&](MCTSNode *node) {
            // Discount the node's score and visits.
            node->result_score *= m_args.node_reuse_discount;
            node->number_of_visits = static_cast<int>(static_cast<float>(node->number_of_visits) *
                                                      m_args.node_reuse_discount);
            for (const NodeId childId : node->children)
                discount(m_pool.get(childId));
        };

        // Discount the node's score and visits.
        discount(root);
    }

    // Run the MCTS iterations until the root node has enough visits.
    while (root->number_of_visits < number_of_searches) {
        parallelIterate(root);
    }

    // Gather the visit counts and result score for the root node.
    VisitCounts visitCounts;
    visitCounts.reserve(root->children.size());
    for (const NodeId childId : root->children) {
        MCTSNode *child = m_pool.get(childId);
        int encodedMove = encodeMove(child->move_to_get_here, &root->board);
        visitCounts.emplace_back(encodedMove, child->number_of_visits);
    }

    if (!root->children.size()) {
        log("Warning: MCTS root node has no children. This is unexpected.");
        log("BoardFen:", root->board.fen());
    }

    return std::tuple{
        MCTSResult(root->result_score / static_cast<float>(root->number_of_visits), visitCounts,
                   root->children),
        mctsStatistics(root, &m_pool),
    };
}
