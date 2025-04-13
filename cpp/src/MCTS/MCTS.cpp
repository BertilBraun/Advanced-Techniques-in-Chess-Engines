#include "MCTS.hpp"

std::vector<MCTSResult> MCTS::search(std::vector<Board> &boards) const {
    if (boards.empty())
        return {};

    TimeItGuard timer("MCTS::search");

    // Get policy moves (with noise) for the given boards.
    const std::vector<std::vector<MoveScore>> movesList = _get_policy_with_noise(boards);

    std::vector<MCTSNode> roots;
    roots.reserve(boards.size());
    for (const auto &[board, moves] : zip(boards, movesList)) {
        // Since no node is pre-expanded, create a new root.
        roots.push_back(MCTSNode::root(board));
        roots.back().expand(moves);
    }

    int iterations = args.num_searches_per_turn / args.num_parallel_searches;
    for (int _ : range(iterations)) {
        parallel_iterate(roots);
    }

    // Build and return the results.
    std::vector<MCTSResult> results;
    results.reserve(roots.size());
    for (const MCTSNode &root : roots) {
        VisitCounts visitCounts;
        visitCounts.reserve(root.children.size());
        for (const MCTSNode &child : root.children) {
            visitCounts.emplace_back(child.encoded_move_to_get_here, child.number_of_visits);
        }
        results.push_back({-root.result_score / (float) root.number_of_visits, visitCounts});
    }

    _logMCTSStatistics(roots);

    return results;
}
void MCTS::parallel_iterate(std::vector<MCTSNode> &roots) const {
    TimeItGuard timer("MCTS::parallel_iterate");

    std::vector<MCTSNode *> nodes;
    nodes.reserve(roots.size() * args.num_parallel_searches);
    for (MCTSNode &root : roots) {
        for (int _ : range(args.num_parallel_searches)) {
            std::optional<MCTSNode *> node = _get_best_child_or_back_propagate(root, args.c_param);
            if (node.has_value()) {
                node.value()->updateVirtualLoss(1);
                nodes.push_back(node.value());
            }
        }
    }
    if (nodes.empty())
        return;

    // Gather boards for inference.
    std::vector<Board> boards;
    boards.reserve(nodes.size());
    for (MCTSNode *node : nodes)
        boards.push_back(node->board);

    // Run inference in batch.
    const std::vector<InferenceResult> results = client->inference_batch(boards);

    for (auto [node, result] : zip(nodes, results)) {
        const auto &[moves, value] = result;
        node->expand(moves);
        node->backPropagate(value);
        node->updateVirtualLoss(-1);
    }
}
std::vector<std::vector<MoveScore>> MCTS::_get_policy_with_noise(std::vector<Board> &boards) const {
    const std::vector<InferenceResult> inferenceResults = client->inference_batch(boards);

    std::vector<std::vector<MoveScore>> noisyMoves;
    noisyMoves.reserve(inferenceResults.size());
    for (auto &result : inferenceResults) {
        noisyMoves.push_back(_add_noise(result.first));
    }
    return noisyMoves;
}

std::vector<MoveScore> MCTS::_add_noise(const std::vector<MoveScore> &moves) const {
    if (moves.empty())
        return {};

    const std::vector<float> noiseList = dirichlet(args.dirichlet_alpha, moves.size());

    std::vector<MoveScore> noisyMoves;
    noisyMoves.reserve(moves.size());
    for (const auto &[noise, moveScore] : zip(noiseList, moves)) {
        const auto &[move, score] = moveScore;
        float newScore = lerp(score, noise, args.dirichlet_epsilon);
        noisyMoves.push_back({move, newScore});
    }

    return noisyMoves;
}

std::optional<MCTSNode *> MCTS::_get_best_child_or_back_propagate(MCTSNode &root,
                                                                  float c_param) const {
    MCTSNode *node = &root;
    while (node->isFullyExpanded()) {
        node = &node->bestChild(c_param);
    }

    if (node->isTerminalNode()) {
        auto result = getBoardResultScore(node->board);
        assert(result.has_value());
        node->backPropagate(result.value());
        return std::nullopt;
    }
    return {node};
}
void MCTS::_logMCTSStatistics(const std::vector<MCTSNode> &roots) const {
    if (roots.empty() || !m_logger)
        return;

    size_t step = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    // Compute the depth of each search tree using a DFS.
    auto dfs = [&](const MCTSNode &node, auto &dfs_ref) -> int {
        if (!node.isFullyExpanded())
            return 0;
        int max_depth = 0;
        for (const MCTSNode &child : node.children) {
            int d = dfs_ref(child, dfs_ref);
            if (d > max_depth)
                max_depth = d;
        }
        return 1 + max_depth;
    };
    std::vector<int> depths;
    for (const MCTSNode &root : roots) {
        depths.push_back(dfs(root, dfs));
    }
    float average_depth = (float) sum(depths) / depths.size();
    m_logger->add_scalar("dataset/average_search_depth", step, average_depth);

    // Compute the entropy of the visit counts.
    auto entropy = [](const MCTSNode &node) -> float {
        int total = node.number_of_visits;
        float ent = 0.0f;
        for (const MCTSNode &child : node.children) {
            if (child.number_of_visits > 0) {
                float p = static_cast<float>(child.number_of_visits) / total;
                ent -= p * std::log2(p);
            }
        }
        return ent;
    };
    std::vector<float> entropies;
    for (const MCTSNode &root : roots) {
        entropies.push_back(entropy(root));
    }
    float average_entropy = sum(entropies) / entropies.size();
    m_logger->add_scalar("dataset/average_search_entropy", step, average_entropy);
}