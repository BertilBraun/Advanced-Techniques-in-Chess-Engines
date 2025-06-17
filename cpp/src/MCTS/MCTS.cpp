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

MCTSResult gatherResult(const MCTSNode *root) {
    VisitCounts visitCounts;
    visitCounts.reserve(root->children.size());
    for (const NodeId childId : root->children) {
        const MCTSNode *child = root->pool->get(childId);
        int encodedMove = encodeMove(child->move_to_get_here, &root->board);
        visitCounts.emplace_back(encodedMove, child->number_of_visits);
    }

    return MCTSResult(root->result_score / static_cast<float>(root->number_of_visits), visitCounts,
                      root->children);
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

    stats.nodePoolCapacity = pool->capacity();
    stats.liveNodeCount = pool->liveNodeCount();

    return stats;
}

std::vector<std::tuple<MCTSResult, MCTSStatistics>>
MCTS::searchGames(const std::vector<BoardTuple> &boards) {
    TIMEIT("MCTS::searchGames");

    std::vector<MCTSNode *> roots;
    roots.reserve(boards.size());

    std::vector<size_t> newBoardIndices;
    std::vector<const Board *> newBoards;
    newBoardIndices.reserve(boards.size());
    newBoards.reserve(boards.size());

    std::cout << "MCTS::searchGames: Preparing " << boards.size() << " boards for search."
              << std::endl;

    for (const auto &[i, board] : enumerate(boards)) {
        const auto &[fen, id, runFullSearch] = board;
        if (id == INVALID_NODE || !m_pool.isLive(id)) {
            std::cout << "MCTS::searchGames: Creating new root node for board " << fen
                      << " (id: " << id << ", runFullSearch: " << runFullSearch << ")."
                      << std::endl;
            MCTSNode *root = m_pool.allocateNode(fen, 1.0, Move::null(), INVALID_NODE, &m_pool);

            roots.push_back(root);

            newBoardIndices.push_back(i);
            newBoards.emplace_back(&root->board);
        } else {
            // If the node is already expanded, we can use it directly.
            // It will have to be cleaned up later. (i.e. its parent and siblings will be removed
            // and visits discounted)
            MCTSNode *root = m_pool.get(id);
            setupNodeForTreeReuse(root, runFullSearch);
            roots.push_back(root);
        }
    }

    std::cout << "MCTS::searchGames: Prepared " << roots.size() << " roots for search."
              << std::endl;

    const std::vector<InferenceResult> inferenceResults = m_client.inferenceBatch(newBoards);

    std::cout << "MCTS::searchGames: Inference completed for " << inferenceResults.size()
              << " boards." << std::endl;

    for (const auto [rootIndex, result] : zip(newBoardIndices, inferenceResults)) {
        const bool shouldRunFullSearch = get<2>(boards[rootIndex]);
        const std::vector<MoveScore> &moves = result.first;

        if (shouldRunFullSearch) {
            // If we are running a full search, we need to add the noise to the root node's policy.
            roots[rootIndex]->expand(addNoise(moves));
        } else {
            // If we are running a fast search, we can use the moves directly.
            roots[rootIndex]->expand(moves);
        }
    }

    std::cout << "MCTS::searchGames: Expanded " << roots.size() << " root nodes." << std::endl;
    std::cout << "MCTS::searchGames: Starting main search loop." << std::endl;

    // -----------------------------------------------------------------
    // 3.  Main search loop – several games in **this** thread.
    // -----------------------------------------------------------------
    std::vector<MCTSNode *> rootsToSearch;
    rootsToSearch.reserve(roots.size());
    while (true) {
        rootsToSearch.clear();
        for (const auto [root, board] : zip(roots, boards)) {
            const auto limit = get<2>(board) ? m_args.num_full_searches : m_args.num_fast_searches;

            if (root->number_of_visits < limit) {
                // If the root has not enough visits, we will search it.
                rootsToSearch.emplace_back(root);
            }
        }

        if (rootsToSearch.empty()) {
            // If there are no roots to search, we can stop.
            break;
        }

        parallelIterate(rootsToSearch);

        std::cout << "MCTS::searchGames: Iterated over " << rootsToSearch.size()
                  << " roots in this iteration." << std::endl;
    }

    std::cout << "MCTS::searchGames: Main search loop finished." << std::endl;

    // -----------------------------------------------------------------
    // 4.  Collect the final best-move + statistics for every root.
    // -----------------------------------------------------------------
    std::vector<std::tuple<MCTSResult, MCTSStatistics>> results;
    results.reserve(roots.size());
    for (const auto *root : roots)
        results.emplace_back(gatherResult(root), mctsStatistics(root, &m_pool));

    std::cout << "MCTS::searchGames: Gathered results for " << results.size() << " roots."
              << std::endl;
    return results;
}

void MCTS::addToNodesToKeep(std::vector<NodeId> &nodesToKeep, const NodeId nodeId) {
    assert(m_pool.isLive(nodeId) &&
           "MCTS::addToNodesToKeep: NodeId is not live, cannot add to nodesToKeep");
    assert(nodeId != INVALID_NODE &&
           "MCTS::addToNodesToKeep: NodeId is invalid, cannot add to nodesToKeep");

    nodesToKeep.push_back(nodeId);

    // Recursively add all children of the node to the list.
    for (const NodeId childId : m_pool.get(nodeId)->children)
        addToNodesToKeep(nodesToKeep, childId);
}

MCTSResults MCTS::search(const std::vector<BoardTuple> &boards) {
    TIMEIT("MCTS::search");

    if (boards.empty())
        return {.results = {}, .mctsStats = {}};

    std::cout << "MCTS::search: Starting search for " << boards.size() << " boards." << std::endl;

    std::vector<NodeId> treeNodesToKeep;

    for (const auto &[_, prevNodeId, ___] : boards) {
        if (prevNodeId != INVALID_NODE)
            addToNodesToKeep(treeNodesToKeep, prevNodeId);
    }

    m_pool.purge(treeNodesToKeep);

    const std::size_t N = boards.size();
    const std::size_t P = std::max<std::size_t>(1, m_threadPool.numThreads());
    const std::size_t sliceSize = (boards.size() + P - 1) / P; // ceiling div

    std::vector<std::future<std::vector<std::tuple<MCTSResult, MCTSStatistics>>>> futures;
    futures.reserve(P);

    for (std::size_t slice = 0; slice < P && slice * sliceSize < boards.size(); ++slice) {
        auto begin = boards.begin() + slice * sliceSize;
        auto end = begin + std::min(sliceSize, boards.size() - slice * sliceSize);
        std::vector<BoardTuple> myBoards(begin, end);

        futures.emplace_back(m_threadPool.enqueue(&MCTS::searchGames, this, std::move(myBoards)));
    }

    std::cout << "MCTS::search: Waiting for " << futures.size() << " futures to complete."
              << std::endl;

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

    std::cout << "MCTS::search: Finished gathering results." << std::endl;

    stats.averageDepth /= static_cast<float>(N);
    stats.averageEntropy /= static_cast<float>(N);
    stats.averageKLDivergence /= static_cast<float>(N);
    stats.nodePoolCapacity = m_pool.capacity();
    stats.liveNodeCount = m_pool.liveNodeCount();

    return {.results = results, .mctsStats = stats};
}

void freeNodeAndChildren(NodePool &pool, const MCTSNode *node, const NodeId excluded) {
    // Free the node and all its children, except the excluded one.
    if (node->myId == excluded)
        return;

    for (const NodeId childId : node->children)
        freeNodeAndChildren(pool, pool.get(childId), excluded);

    pool.deallocateNode(node->myId);
}

void MCTS::freeTree(const NodeId nodeId, const NodeId excluded) {
    // Free the node, its parent and all its children, except the excluded one and its children.
    const MCTSNode *root = m_pool.get(nodeId);
    while (root->parent != INVALID_NODE)
        // If the root has a parent, we need to walk up the tree to find the root node.
        root = m_pool.get(root->parent);

    freeNodeAndChildren(m_pool, root, excluded);
}

MCTSResult MCTS::evalSearch(const std::string &fen, const NodeId prevNodeId,
                            const int numberOfSearches) {
    MCTSNode *root;
    if (prevNodeId == INVALID_NODE || !m_pool.isLive(prevNodeId)) {
        // If the previous node is invalid or not live, we need to create a new root node.
        root = m_pool.allocateNode(fen, 1.0, Move::null(), INVALID_NODE, &m_pool);
    } else {
        // If the previous node is valid, we can use it as the root node.
        root = m_pool.get(prevNodeId);
        assert(root->board.fen() == fen &&
               "MCTS::evalSearch: Previous node's board FEN does not match the given FEN");
    }

    // TODO make use of num_threads parallel inference threads -> requires mutexes in MCTSNode

    std::vector<MCTSNode *> roots;
    roots.push_back(root);

    // Run the MCTS iterations until the root node has enough visits.
    for (int _ : range(numberOfSearches))
        parallelIterate(roots);

    return gatherResult(root);
}

void MCTS::parallelIterate(const std::vector<MCTSNode *> &roots) {
    TIMEIT("MCTS::parallelIterate");
    // These variables are initialized only once per thread
    // and retain their values between function calls
    thread_local std::vector<MCTSNode *> nodes;
    thread_local std::vector<const Board *> boards;

    // Clear contents but maintain capacity
    nodes.clear();
    boards.clear();
    nodes.reserve(m_args.num_parallel_searches * roots.size());
    boards.reserve(m_args.num_parallel_searches * roots.size());

    std::cout << "MCTS::parallelIterate: Starting parallel iteration for " << roots.size()
              << " roots." << std::endl;

    for (MCTSNode *root : roots) {
        if (root->myId == root->parent) {
            throw std::runtime_error("MCTS::parallelIterate: The root node must have a parent. "
                                     "This is likely a bug in the MCTS implementation.");
        }
        std::cout << "MCTS::parallelIterate: Processing root node: " << root->repr() << std::endl;
        for (int _ : range(m_args.num_parallel_searches)) {
            MCTSNode *node = getBestChildOrBackPropagate(root, m_args.c_param);
            if (node != nullptr) {
                std::cout << "MCTS::parallelIterate: Processing node: " << node->repr()
                          << std::endl;
                node->addVirtualLoss();
                std::cout << "MCTS::parallelIterate: Added virtual loss to node: " << node->repr()
                          << std::endl;
                nodes.push_back(node);
                boards.emplace_back(&node->board);
            }
        }
    }

    std::cout << "MCTS::parallelIterate: Prepared " << nodes.size()
              << " nodes for parallel inference." << std::endl;

    if (nodes.empty())
        return;

    // Run inference in batch.
    const std::vector<InferenceResult> results = m_client.inferenceBatch(boards);

    std::cout << "MCTS::parallelIterate: Inference completed for " << results.size() << " nodes."
              << std::endl;

    for (auto [node, result] : zip(nodes, results)) {
        const auto &[moves, value] = result;
        std::cout << "MCTS::parallelIterate: Expanding node: " << node->repr()
                  << " with moves size: " << moves.size() << " and value: " << value << std::endl;
        node->expand(moves);
        std::cout << "MCTS::parallelIterate: Back propagating value: " << value
                  << " for node: " << node->repr() << std::endl;
        node->backPropagateAndRemoveVirtualLoss(value);
        std::cout << "MCTS::parallelIterate: Back propagation completed for node: " << node->repr()
                  << std::endl;
    }

    std::cout << "MCTS::parallelIterate: Finished parallel iteration for " << roots.size()
              << " roots." << std::endl;
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
MCTSNode *MCTS::getBestChildOrBackPropagate(MCTSNode *root, const float cParam) {
    TIMEIT("MCTS::getBestChildOrBackPropagate");

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

void MCTS::setupNodeForTreeReuse(MCTSNode *root, const bool shouldRunFullSearch) {
    assert(root->parent != INVALID_NODE && "MCTS::setupRoot: The root node must have a parent");
    // If the node has a parent, we need to clean it up first.
    // Remove the nodes parent and all its children from the pool.
    root->parent = INVALID_NODE;

    // Add Dirichlet noise to the node's policy only if it's a full search, don't add noise for
    // fast searches.
    if (shouldRunFullSearch) {
        const std::vector<float> noise = dirichlet(m_args.dirichlet_alpha, root->children.size());

        for (const auto [i, childId] : enumerate(root->children)) {
            MCTSNode *child = m_pool.get(childId);
            child->policy = lerp(child->policy, noise[i], m_args.dirichlet_epsilon);
        }
    }

    std::function<void(MCTSNode *)> discount = [&](MCTSNode *node) {
        // Discount the node's score and visits. - Problem same divisor is not given because of
        // integer rounding
        node->number_of_visits = static_cast<int>(static_cast<float>(node->number_of_visits) *
                                                  m_args.node_reuse_discount);
        node->result_score = static_cast<float>(static_cast<int>(
            static_cast<float>(static_cast<int>(node->result_score)) * m_args.node_reuse_discount));

        node->result_score = clamp(node->result_score, static_cast<float>(-node->number_of_visits),
                                   static_cast<float>(node->number_of_visits));

        for (const NodeId childId : node->children)
            discount(m_pool.get(childId));
    };

    // Discount the node's score and visits.
    discount(root);
}
