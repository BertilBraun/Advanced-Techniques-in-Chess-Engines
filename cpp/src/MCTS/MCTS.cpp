#include "MCTS.hpp"

#include "../BoardEncoding.hpp"

#include <ranges>

thread_local std::mt19937 randomEngine(std::random_device{}());

std::vector<float> dirichlet(float alpha, size_t n) {
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

// Traverse the tree to find the best child or, if the node is terminal,
// back-propagate the board’s result.
std::optional<MCTSNode *> getBestChildOrBackPropagate(MCTSNode &root, float c_param) {
    MCTSNode *node = &root;
    while (node->isFullyExpanded()) {
        node = &node->bestChild(c_param);
    }

    if (node->isTerminalNode()) {
        const auto result = getBoardResultScore(node->board);
        assert(result.has_value());
        node->backPropagate(result.value());
        return std::nullopt;
    }
    return {node};
}

MCTSStatistics mctsStatistics(const std::vector<MCTSNode> &roots) {
    MCTSStatistics stats;
    if (roots.empty())
        return stats;

    {
        // Compute the depth of each search tree using a DFS.
        auto dfs = [&](const MCTSNode &node, auto &dfs_ref) -> int {
            if (!node.isFullyExpanded())
                return 0;
            int maxDepth = 0;
            for (const MCTSNode &child : node.children) {
                const int d = dfs_ref(child, dfs_ref);
                if (d > maxDepth)
                    maxDepth = d;
            }
            return 1 + maxDepth;
        };
        std::vector<int> depths;
        depths.reserve(roots.size());
        for (const MCTSNode &root : roots) {
            depths.push_back(dfs(root, dfs));
        }
        stats.averageDepth = static_cast<float>(sum(depths)) / depths.size();
    }
    {
        // Compute the entropy of the visit counts.
        auto entropy = [](const MCTSNode &node) -> float {
            const int total = node.number_of_visits;
            float ent = 0.0f;
            for (const MCTSNode &child : node.children) {
                if (child.number_of_visits > 0) {
                    const float p = static_cast<float>(child.number_of_visits) / total;
                    ent -= p * std::log2(p);
                }
            }
            return ent;
        };

        std::vector<float> entropies;
        entropies.reserve(roots.size());
        for (const MCTSNode &root : roots) {
            entropies.push_back(entropy(root));
        }
        stats.averageEntropy = sum(entropies) / entropies.size();
    }
    {
        auto klDivergence = [](const MCTSNode &node) -> float {
            const int total = node.number_of_visits;
            float kl = 0.0f;
            for (const MCTSNode &child : node.children) {
                if (child.number_of_visits > 0) {
                    const float p = static_cast<float>(child.number_of_visits) / total;
                    const float q = 1.0f / node.children.size(); // Uniform distribution
                    if (p > 0.0f)
                        kl += p * std::log2(p / q);
                }
            }
            return kl;
        };

        std::vector<float> klDivergences;
        klDivergences.reserve(roots.size());
        for (const MCTSNode &root : roots) {
            klDivergences.push_back(klDivergence(root));
        }
        stats.averageKLDivergence = sum(klDivergences) / klDivergences.size();
    }

    return stats;
}

MCTSResults MCTS::search(std::vector<std::tuple<std::string, MCTSNode, bool>> &boards) {
    if (boards.empty())
        return {.results = {}, .mctsStats = {}};

    // Get policy moves (with noise) for the given boards.
    const std::vector<std::vector<MoveScore>> movesList = getPolicyWithNoise(boards);

    std::vector<MCTSNode> roots;
    roots.reserve(boards.size());
    for (const auto &[board, moves] : zip(boards, movesList)) {
        // Since no node is pre-expanded, create a new root.
        roots.push_back(MCTSNode::root(board));
        roots.back().expand(moves);
    }

    int iterations = m_args.num_searches_per_turn / m_args.num_parallel_searches;
    for (int _ : range(iterations)) {
        parallelIterate(roots);
    }

    // Build and return the results.
    std::vector<MCTSResult> results;
    results.reserve(roots.size());
    for (const MCTSNode &root : roots) {
        VisitCounts visitCounts;
        visitCounts.reserve(root.children.size());
        for (const MCTSNode &child : root.children) {
            visitCounts.emplace_back(child.move_to_get_here, child.number_of_visits);
        }
        results.emplace_back(root.result_score / static_cast<float>(root.number_of_visits),
                             visitCounts, root.children, // TODO
        );
    }

    return {
        .results = results,
        .mctsStats = mctsStatistics(roots),
    };
}

InferenceStatistics MCTS::getInferenceStatistics() const { return m_client.getStatistics(); }

void MCTS::parallelIterate(std::vector<MCTSNode> &roots) {
    std::vector<MCTSNode *> nodes;
    nodes.reserve(roots.size() * m_args.num_parallel_searches);
    for (MCTSNode &root : roots) {
        for (int _ : range(m_args.num_parallel_searches)) {
            std::optional<MCTSNode *> node = getBestChildOrBackPropagate(root, m_args.c_param);
            if (node.has_value()) {
                node.value()->updateVirtualLoss(1);
                nodes.push_back(node.value());
            }
        }
    }
    if (nodes.empty())
        return;

    // Gather boards for inference.
    std::vector<const Board &> boards;
    boards.reserve(nodes.size());
    for (const MCTSNode *node : nodes)
        boards.push_back(node->board);

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
MCTS::getPolicyWithNoise(const std::vector<const Board &> &boards) {
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
