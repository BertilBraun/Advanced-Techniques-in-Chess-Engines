#include "MCTS.hpp"

#include "../BoardEncoding.hpp"
#include "MoveEncoding.hpp"

thread_local std::mt19937 randomEngine(std::random_device{}());

namespace {
struct RootTask {
    MCTSRoot root;
    uint32 limit;
};

struct SelectedNode {
    SearchTree *tree;
    NodeIndex index;
};

thread_local std::vector<SelectedNode> selectedNodes;
thread_local std::vector<const Board *> selectedBoards;

std::vector<float> dirichlet(const float alpha, const size_t count) {
    std::gamma_distribution<float> gamma(alpha, 1.0);

    std::vector<float> noise(count);
    float sum = 0.0f;
    for (float &sample : noise) {
        sample = gamma(randomEngine);
        sum += sample;
    }

    const float normalization = 1.0f / sum;
    for (float &sample : noise) {
        sample *= normalization;
    }
    return noise;
}

MCTSResult gatherResult(const MCTSRoot &root) {
    const SearchNode &rootNode = root.tree().node(root.rootIndex());
    VisitCounts visitCounts;
    visitCounts.reserve(rootNode.children.size());
    for (const Child &child : rootNode.children) {
        visitCounts.emplace_back(encodeMove(child.move, &rootNode.board),
                                 static_cast<int>(child.number_of_visits));
    }

    return MCTSResult(root.resultSum() / static_cast<float>(root.visits()), visitCounts, root);
}

MCTSStatistics mctsStatistics(const MCTSRoot &root) {
    MCTSStatistics statistics;
    statistics.averageDepth = static_cast<float>(root.maxDepth());
    const SearchNode &rootNode = root.tree().node(root.rootIndex());
    if (rootNode.children.empty()) {
        return statistics;
    }

    const float totalVisits = static_cast<float>(root.visits());
    const float uniformProbability = 1.0f / static_cast<float>(rootNode.children.size());
    for (const Child &child : rootNode.children) {
        if (child.number_of_visits == 0) {
            continue;
        }

        const float probability = static_cast<float>(child.number_of_visits) / totalVisits;
        statistics.averageEntropy -= probability * std::log2(probability);
        statistics.averageKLDivergence += probability * std::log2(probability / uniformProbability);
    }
    return statistics;
}
} // namespace

MCTSParams::MCTSParams(const int numParallelSearches, const uint32 numFullSearches,
                       const uint32 numFastSearches, const float cParam, const float dirichletAlpha,
                       const float dirichletEpsilon, const uint8 minVisitCount,
                       const uint8 numThreads)
    : num_parallel_searches(numParallelSearches), num_full_searches(numFullSearches),
      num_fast_searches(numFastSearches), c_param(cParam), dirichlet_alpha(dirichletAlpha),
      dirichlet_epsilon(dirichletEpsilon), min_visit_count(minVisitCount), num_threads(numThreads) {
    if (num_parallel_searches <= 0) {
        throw std::invalid_argument("num_parallel_searches must be positive");
    }
    if (num_full_searches == 0 || num_fast_searches == 0) {
        throw std::invalid_argument("MCTS search budgets must be positive");
    }
    if (num_threads == 0) {
        throw std::invalid_argument("num_threads must be positive");
    }
}

uint32 MCTSParams::arenaCapacity() const {
    const uint64 maximumSearches = std::max(num_full_searches, num_fast_searches);
    const uint64 capacity = maximumSearches + static_cast<uint64>(num_parallel_searches) + 1U;
    if (capacity > std::numeric_limits<uint32>::max()) {
        throw std::overflow_error("MCTS search parameters exceed the node index capacity");
    }
    return static_cast<uint32>(capacity);
}

MCTSRoot MCTS::newRoot(const std::string &fen) const {
    return MCTSRoot::create(fen, m_arenaCapacity);
}

MCTSRoot MCTS::newRoot(Board board) const {
    return MCTSRoot::create(std::move(board), m_arenaCapacity);
}

std::vector<MCTSResult> MCTS::searchGames(const std::vector<BoardTuple> &boards) {
    TIMEIT("MCTS::searchGames");

    const size_t numberOfBoards = boards.size();
    std::vector<MCTSRoot> roots;
    roots.reserve(numberOfBoards);

    std::vector<size_t> newBoardIndices;
    std::vector<const Board *> newBoards;
    newBoardIndices.reserve(numberOfBoards);
    newBoards.reserve(numberOfBoards);

    for (const auto &[index, board] : enumerate(boards)) {
        const auto &[root, runFullSearch] = board;
        static_cast<void>(runFullSearch);
        if (root.arenaCapacity() != m_arenaCapacity) {
            throw std::invalid_argument("MCTSRoot arena capacity does not match MCTS parameters");
        }
        roots.push_back(root);

        if (!root.isExpanded()) {
            newBoardIndices.emplace_back(index);
            newBoards.emplace_back(&root.board());
        }
    }

    const std::vector<InferenceResult> inferenceResults = m_client.inferenceBatch(newBoards);
    for (const auto [rootIndex, result] : zip(newBoardIndices, inferenceResults)) {
        roots[rootIndex].tree().expand(roots[rootIndex].rootIndex(), result.first);
    }

    for (size_t index = 0; index < roots.size(); ++index) {
        if (get<1>(boards[index])) {
            addNoise(roots[index]);
        }
    }

    std::vector<RootTask> active;
    active.reserve(numberOfBoards);
    for (size_t index = 0; index < numberOfBoards; ++index) {
        const uint32 limit =
            get<1>(boards[index]) ? m_args.num_full_searches : m_args.num_fast_searches;
        active.push_back({roots[index], limit});
    }

    while (!active.empty()) {
        std::vector<MCTSRoot> batch;
        batch.reserve(active.size());
        for (const RootTask &task : active) {
            batch.push_back(task.root);
        }

        parallelIterate(batch);
        std::erase_if(active,
                      [](const RootTask &task) { return task.root.visits() >= task.limit; });
    }

    std::vector<MCTSResult> results;
    results.reserve(roots.size());
    for (const MCTSRoot &root : roots) {
        results.emplace_back(gatherResult(root));
    }
    return results;
}

MCTSResults MCTS::search(const std::vector<BoardTuple> &boards, const bool collectStatistics) {
    TIMEIT("MCTS::search");
    if (boards.empty()) {
        return {.results = {}, .mctsStats = {}};
    }

    const size_t numberOfBoards = boards.size();
    const size_t numberOfThreads = std::max<size_t>(1, m_threadPool.numThreads());
    const size_t sliceSize = (numberOfBoards + numberOfThreads - 1) / numberOfThreads;

    std::vector<std::future<std::vector<MCTSResult>>> futures;
    futures.reserve(numberOfThreads);
    for (size_t slice = 0; slice < numberOfThreads && slice * sliceSize < numberOfBoards; ++slice) {
        const auto begin = boards.begin() + static_cast<std::ptrdiff_t>(slice * sliceSize);
        const auto end = begin + static_cast<std::ptrdiff_t>(
                                     std::min(sliceSize, numberOfBoards - slice * sliceSize));
        std::vector<BoardTuple> slicedBoards(begin, end);
        futures.emplace_back(
            m_threadPool.enqueue(&MCTS::searchGames, this, std::move(slicedBoards)));
    }

    std::vector<MCTSResult> results;
    results.reserve(numberOfBoards);
    for (std::future<std::vector<MCTSResult>> &future : futures) {
        std::vector<MCTSResult> sliceResults = future.get();
        results.insert(results.end(), std::make_move_iterator(sliceResults.begin()),
                       std::make_move_iterator(sliceResults.end()));
    }

    const MCTSStatistics statistics =
        collectStatistics ? mctsStatistics(results.front().root) : MCTSStatistics{};
    return {.results = std::move(results), .mctsStats = statistics};
}

std::pair<InferenceStatistics, TimeInfo> MCTS::getInferenceStatistics() {
    return {m_client.getStatistics(), resetTimes()};
}

void MCTS::parallelIterate(const std::vector<MCTSRoot> &roots) {
    TIMEIT("MCTS::parallelIterate");
    selectedNodes.clear();
    selectedBoards.clear();
    const size_t selectionCapacity =
        static_cast<size_t>(m_args.num_parallel_searches) * roots.size();
    selectedNodes.reserve(selectionCapacity);
    selectedBoards.reserve(selectionCapacity);

    for (MCTSRoot root : roots) {
        for (int search = 0; search < m_args.num_parallel_searches; ++search) {
            const std::optional<NodeIndex> selected =
                getBestChildOrBackPropagate(root, m_args.c_param);
            if (selected.has_value()) {
                SearchTree &tree = root.tree();
                tree.addVirtualLoss(*selected);
                selectedNodes.push_back({&tree, *selected});
                selectedBoards.push_back(&tree.node(*selected).board);
            }
        }
    }

    if (selectedNodes.empty()) {
        return;
    }

    const std::vector<InferenceResult> results = m_client.inferenceBatch(selectedBoards);
    for (const auto [selected, result] : zip(selectedNodes, results)) {
        const auto &[moves, value] = result;
        selected.tree->expand(selected.index, moves);
        selected.tree->backPropagateAndRemoveVirtualLoss(selected.index, value);
    }
}

void MCTS::addNoise(MCTSRoot &root) const {
    SearchNode &rootNode = root.tree().node(root.rootIndex());
    const std::vector<float> noise = dirichlet(m_args.dirichlet_alpha, rootNode.children.size());
    for (size_t index = 0; index < rootNode.children.size(); ++index) {
        Child &child = rootNode.children[index];
        child.policy = lerp(child.policy, noise[index], m_args.dirichlet_epsilon);
    }
}

std::optional<NodeIndex> MCTS::getBestChildOrBackPropagate(MCTSRoot &root,
                                                           const float cParam) const {
    TIMEIT("MCTS::getBestChildOrBackPropagate");
    SearchTree &tree = root.tree();
    NodeIndex selectedIndex = root.rootIndex();
    SearchNode &rootNode = tree.node(selectedIndex);
    for (uint32 childIndex = 0; childIndex < rootNode.children.size(); ++childIndex) {
        if (rootNode.children[childIndex].number_of_visits < m_args.min_visit_count) {
            selectedIndex = tree.materializeChild(selectedIndex, childIndex);
            break;
        }
    }

    while (tree.node(selectedIndex).isExpanded()) {
        const uint32 childIndex = tree.bestChildIndex(selectedIndex, cParam);
        selectedIndex = tree.materializeChild(selectedIndex, childIndex);
    }

    if (tree.node(selectedIndex).isTerminal()) {
        tree.backPropagate(selectedIndex, getBoardResultScore(tree.node(selectedIndex).board));
        return std::nullopt;
    }
    return selectedIndex;
}
