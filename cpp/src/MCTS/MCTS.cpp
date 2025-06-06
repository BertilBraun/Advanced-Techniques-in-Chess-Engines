#include "MCTS.hpp"

#include "../BoardEncoding.hpp"

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

    float norm = 1.0f / sum;
    for (size_t i = 0; i < n; i++) {
        noise[i] *= norm;
    }

    return noise;
}

MCTSStatistics mctsStatistics(const std::vector<MCTSNode *> &roots, NodePool *pool) {
    MCTSStatistics stats;
    if (roots.empty())
        return stats;

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
        std::vector<int> depths;
        depths.reserve(roots.size());
        for (const MCTSNode *root : roots) {
            depths.push_back(dfs(root));
        }
        stats.averageDepth = static_cast<float>(sum(depths)) / depths.size();
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

        std::vector<float> entropies;
        entropies.reserve(roots.size());
        for (const MCTSNode *root : roots) {
            entropies.push_back(entropy(root));
        }
        stats.averageEntropy = sum(entropies) / entropies.size();
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

        std::vector<float> klDivergences;
        klDivergences.reserve(roots.size());
        for (const MCTSNode *root : roots) {
            klDivergences.push_back(klDivergence(root));
        }
        stats.averageKLDivergence = sum(klDivergences) / klDivergences.size();
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
            const NodeId newId = m_pool.allocateNode(fen, 1.0, Move::null(), INVALID_NODE, &m_pool);
            MCTSNode *root = m_pool.get(newId);
            root->myId = newId;
            roots.push_back(root);
            newBoards.emplace_back(&root->board);
        } else {
            // If the node is already expanded, we can use it directly.

            std::function<void(NodeId, NodeId)> free = [&](const NodeId root,
                                                           const NodeId excluded) {
                // Free the node and all its children, except the excluded one.
                if (root == excluded)
                    return;

                for (const NodeId childId : m_pool.get(root)->children)
                    free(childId, excluded);

                m_pool.deallocateNode(root);
            };

            MCTSNode *root = m_pool.get(id);

            // Remove the nodes parent and all its children from the pool.
            free(root->parent, root->myId);
            root->parent = INVALID_NODE;

            // Add Dirichlet noise to the node's policy.
            const std::vector<float> noise =
                dirichlet(m_args.dirichlet_alpha, root->children.size());

            for (const auto [i, childId] : enumerate(root->children)) {
                MCTSNode *child = m_pool.get(childId);
                child->policy = lerp(child->policy, noise[i], m_args.dirichlet_epsilon);
            }

            std::function<void(MCTSNode *)> discount = [&](MCTSNode *node) {
                // Discount the node's score and visits.
                node->result_score *= m_args.node_reuse_discount;
                node->number_of_visits *= m_args.node_reuse_discount;
                for (const NodeId childId : node->children)
                    discount(m_pool.get(childId));
            };

            // Discount the node's score and visits.
            discount(root);

            roots.push_back(root);
        }
    }

    // Get policy moves (with noise) for the given boards.
    const std::vector<std::vector<MoveScore>> movesList = getPolicyWithNoise(newBoards);

    size_t moveIndex = 0;

    for (const auto root : roots) {
        if (!root->isFullyExpanded()) {
            // If the node is not pre-expanded, we need to expand it with the moves.
            root->expand(movesList[moveIndex]);

            moveIndex++;
        }
    }

    const std::size_t N = boards.size();
    std::vector<std::future<MCTSResult>> futures;
    futures.reserve(N);

    for (const auto &[root, tup] : zip(roots, boards))
        futures.emplace_back(m_threadPool.enqueue(
            [this](MCTSNode *node, int number_of_searches) {
                // Run the MCTS iterations until the root node has enough visits.
                while (node->number_of_visits < number_of_searches) {
                    parallelIterate(node);
                }

                // Gather the visit counts and result score for the root node.
                VisitCounts visitCounts;
                visitCounts.reserve(node->children.size());
                for (const NodeId childId : node->children) {
                    MCTSNode *child = m_pool.get(childId);
                    visitCounts.emplace_back(child->move_to_get_here, child->number_of_visits);
                }

                return MCTSResult(node->result_score / static_cast<float>(node->number_of_visits),
                                  visitCounts, node->children);
            },
            root, get<2>(tup)));

    std::vector<MCTSResult> results;
    results.reserve(N);
    for (auto &fut : futures)
        results.emplace_back(fut.get());

    return {
        .results = results,
        .mctsStats = mctsStatistics(roots, &m_pool),
    };
}

void MCTS::parallelIterate(const MCTSNode *root) {
    std::vector<MCTSNode *> nodes;
    nodes.reserve(m_args.num_parallel_searches);
    for (int _ : range(m_args.num_parallel_searches)) {
        std::optional<MCTSNode *> node = getBestChildOrBackPropagate(root, m_args.c_param);
        if (node.has_value()) {
            node.value()->updateVirtualLoss(1);
            nodes.push_back(node.value());
        }
    }
    if (nodes.empty())
        return;

    // Gather boards for inference.
    std::vector<const Board *> boards;
    boards.reserve(nodes.size());
    for (const MCTSNode *node : nodes)
        boards.emplace_back(&node->board);

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
        noisyMoves.push_back({move, newScore});
    }

    return noisyMoves;
}

// Traverse the tree to find the best child or, if the node is terminal,
// back-propagate the board’s result.
std::optional<MCTSNode *> MCTS::getBestChildOrBackPropagate(const MCTSNode *root,
                                                            const float cParam) {

    MCTSNode *finalNode = nullptr;
    for (const NodeId childId : root->children) {
        MCTSNode *child = m_pool.get(childId);
        if (child->number_of_visits < m_args.min_visit_count) {
            // If the child has not been visited enough, we can return it directly.
            finalNode = child;
            break;
        }
    }

    if (finalNode == nullptr) {
        NodeId nodeId = root->myId;
        while (true) {
            const MCTSNode *node = m_pool.get(nodeId);

            if (!node->isFullyExpanded())
                break;

            nodeId = node->bestChild(cParam);
        }

        finalNode = m_pool.get(nodeId);
    }

    if (finalNode->isTerminalNode()) {
        const auto result = getBoardResultScore(finalNode->board);
        assert(result.has_value());
        finalNode->backPropagate(result.value());
        return std::nullopt;
    }
    return {finalNode};
}
