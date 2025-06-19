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

MCTSResult gatherResult(const std::shared_ptr<MCTSNode> &root) {
    VisitCounts visitCounts;
    visitCounts.reserve(root->children.size());
    for (const auto &child : root->children) {
        int encodedMove = encodeMove(child->move_to_get_here, &root->board);
        visitCounts.emplace_back(encodedMove, child->number_of_visits);
    }

    return MCTSResult(root->result_sum / static_cast<float>(root->number_of_visits), visitCounts,
                      root);
}

MCTSStatistics mctsStatistics(const std::shared_ptr<MCTSNode> &root) {
    MCTSStatistics stats;

    {
        // Compute the depth of each search tree using a DFS.
        stats.averageDepth = static_cast<float>(root->maxDepth());
    }
    {
        // Compute the entropy of the visit counts.
        auto entropy = [&](const std::shared_ptr<MCTSNode> &node) -> float {
            const int total = node->number_of_visits;
            float ent = 0.0f;
            for (const auto &child : node->children) {
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
        auto klDivergence = [&](const std::shared_ptr<MCTSNode> &node) -> float {
            const int total = node->number_of_visits;
            float kl = 0.0f;
            for (const auto &child : node->children) {
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

struct RootTask {
    std::shared_ptr<MCTSNode> root;
    uint32 limit; // visit budget for this root
};

std::vector<std::pair<MCTSResult, MCTSStatistics>>
MCTS::searchGames(const std::vector<BoardTuple> &boards) {
    TIMEIT("MCTS::searchGames");

    const size_t N = boards.size();

    std::vector<std::shared_ptr<MCTSNode>> roots;
    roots.reserve(N);

    std::vector<size_t> newBoardIndices;
    std::vector<const Board *> newBoards;
    newBoardIndices.reserve(N);
    newBoards.reserve(N);

    for (const auto &[i, board] : enumerate(boards)) {
        const auto &[root, runFullSearch] = board;

        assert(!root->parent.lock() && "Root node should not have a parent");

        roots.push_back(root);

        if (!root->isExpanded()) {
            newBoardIndices.emplace_back(i);
            newBoards.emplace_back(&root->board);
        }
    }

    const std::vector<InferenceResult> inferenceResults = m_client.inferenceBatch(newBoards);

    for (const auto [rootIndex, result] : zip(newBoardIndices, inferenceResults)) {
        const std::vector<MoveScore> &moves = result.first;
        roots[rootIndex]->expand(moves);
    }

    for (const auto &[root, board] : zip(roots, boards)) {
        if (const bool shouldRunFullSearch = get<1>(board)) {
            // If we are running a full search, we need to add the noise to the root node's policy.
            addNoise(root);
        }
    }

    // -----------------------------------------------------------------
    // Pre-compute the per-root visit limit and build the active list
    // -----------------------------------------------------------------

    std::vector<RootTask> active;
    active.reserve(N);

    for (const int i : range(N)) {
        const uint32 limit =
            get<1>(boards[i]) ? m_args.num_full_searches : m_args.num_fast_searches;
        active.emplace_back(roots[i], limit);
    }

    // -----------------------------------------------------------------
    // Main search loop – continues until every root reaches its limit
    // -----------------------------------------------------------------
    while (!active.empty()) {

        // Expose just the raw node pointers to the parallel routine.
        std::vector<std::shared_ptr<MCTSNode>> batch;
        batch.reserve(active.size());
        for (auto &[root, limit] : active)
            batch.emplace_back(root);

        parallelIterate(batch);

        // Keep only the roots that still need work.
        std::erase_if(active,
                      [](const RootTask &t) { return t.root->number_of_visits >= t.limit; });
    }

    // -----------------------------------------------------------------
    // Collect the final best-move + statistics for every root.
    // -----------------------------------------------------------------
    std::vector<std::pair<MCTSResult, MCTSStatistics>> results;
    results.reserve(roots.size());
    for (const auto &root : roots)
        results.emplace_back(gatherResult(root), mctsStatistics(root));

    return results;
}

MCTSResults MCTS::search(const std::vector<BoardTuple> &boards) {
    TIMEIT("MCTS::search");

    if (boards.empty())
        return {.results = {}, .mctsStats = {}};

    const size_t N = boards.size();
    const size_t P = std::max<size_t>(1, m_threadPool.numThreads());
    const size_t sliceSize = (boards.size() + P - 1) / P; // ceiling div

    std::vector<std::future<std::vector<std::pair<MCTSResult, MCTSStatistics>>>> futures;
    futures.reserve(P);

    for (std::size_t slice = 0; slice < P && slice * sliceSize < boards.size(); ++slice) {
        auto begin = boards.begin() + slice * sliceSize;
        auto end = begin + std::min(sliceSize, boards.size() - slice * sliceSize);
        std::vector<BoardTuple> myBoards(begin, end);

        futures.emplace_back(m_threadPool.enqueue(&MCTS::searchGames, this, std::move(myBoards)));
    }

    std::vector<MCTSResult> results;
    MCTSStatistics stats;
    results.reserve(N);
    for (auto &fut : futures) {
        for (const auto &[result, rootStats] : fut.get()) {
            stats.averageDepth += rootStats.averageDepth;
            stats.averageEntropy += rootStats.averageEntropy;
            stats.averageKLDivergence += rootStats.averageKLDivergence;

            results.emplace_back(result);
        }
    }

    stats.averageDepth /= static_cast<float>(N);
    stats.averageEntropy /= static_cast<float>(N);
    stats.averageKLDivergence /= static_cast<float>(N);

    return {.results = results, .mctsStats = stats};
}

std::pair<InferenceStatistics, TimeInfo> MCTS::getInferenceStatistics() {
    return {m_client.getStatistics(), resetTimes()};
}

MCTSResult MCTS::evalSearch(const std::shared_ptr<MCTSNode> &root, const int numberOfSearches) {
    // TODO make use of num_threads parallel inference threads -> requires mutexes in MCTSNode

    std::vector<std::shared_ptr<MCTSNode>> roots;
    roots.push_back(root);

    // Run the MCTS iterations until the root node has enough visits.
    for (int _ : range(numberOfSearches))
        parallelIterate(roots);

    return gatherResult(root);
}

// These variables are initialized only once per thread
// and retain their values between function calls
thread_local std::vector<std::shared_ptr<MCTSNode>> nodes;
thread_local std::vector<const Board *> boards;

void MCTS::parallelIterate(const std::vector<std::shared_ptr<MCTSNode>> &roots) {
    TIMEIT("MCTS::parallelIterate");

    // Clear contents but maintain capacity
    nodes.clear();
    boards.clear();
    nodes.reserve(m_args.num_parallel_searches * roots.size());
    boards.reserve(m_args.num_parallel_searches * roots.size());

    for (const std::shared_ptr<MCTSNode> &root : roots) {
        for (int _ : range(m_args.num_parallel_searches)) {
            std::shared_ptr<MCTSNode> node = getBestChildOrBackPropagate(root, m_args.c_param);
            if (node != nullptr) {
                node->addVirtualLoss();
                nodes.push_back(node);
                boards.emplace_back(&node->board);
            }
        }
    }

    if (nodes.empty())
        return;

    // Run inference in batch.
    const std::vector<InferenceResult> results = m_client.inferenceBatch(boards);

    for (auto [node, result] : zip(nodes, results)) {
        const auto &[moves, value] = result;
        node->expand(moves);
        node->backPropagateAndRemoveVirtualLoss(value);
    }
}

void MCTS::addNoise(const std::shared_ptr<MCTSNode> &root) const {

    const std::vector<float> noise = dirichlet(m_args.dirichlet_alpha, root->children.size());

    for (const auto &[i, child] : enumerate(root->children)) {
        child->policy = lerp(child->policy, noise[i], m_args.dirichlet_epsilon);
    }
}

// Traverse the tree to find the best child or, if the node is terminal,
// back-propagate the board’s result.
std::shared_ptr<MCTSNode> MCTS::getBestChildOrBackPropagate(std::shared_ptr<MCTSNode> root,
                                                            const float cParam) const {
    TIMEIT("MCTS::getBestChildOrBackPropagate");

    for (const auto &child : root->children) {
        if (child->number_of_visits < m_args.min_visit_count) {
            // If the child has not been visited enough, we should traverse it.
            root = child;
            break;
        }
    }

    // We need to traverse the tree until we find a node that is not fully expanded
    std::shared_ptr<MCTSNode> node = root;
    while (node->isExpanded())
        node = node->bestChild(cParam);

    if (node->isTerminal()) {
        node->backPropagate(getBoardResultScore(node->board));
        return nullptr;
    }
    return node;
}
