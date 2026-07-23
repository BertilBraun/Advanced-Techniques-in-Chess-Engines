#include "MCTS.hpp"

#include "../BoardEncoding.hpp"
#include "DirectSelfPlaySearch.hpp"
#include "MoveEncoding.hpp"

#include <numeric>

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

DirectSelfPlayInferenceParams::DirectSelfPlayInferenceParams(const int inferenceWorkers,
                                                             const int inferenceBatchSize,
                                                             const int outstandingBatchesPerWorker)
    : inference_workers(inferenceWorkers), inference_batch_size(inferenceBatchSize),
      outstanding_batches_per_worker(outstandingBatchesPerWorker) {
    if (inference_workers <= 0) {
        throw std::invalid_argument("inference_workers must be positive");
    }
    if (inference_batch_size <= 0) {
        throw std::invalid_argument("inference_batch_size must be positive");
    }
    if (outstanding_batches_per_worker <= 0 || outstanding_batches_per_worker > 2) {
        throw std::invalid_argument("outstanding_batches_per_worker must be 1 or 2");
    }
}

MCTS::MCTS(const InferenceClientParams &clientArgs, const MCTSParams &mctsArgs,
           const bool useInferenceCache,
           std::optional<DirectSelfPlayInferenceParams> directInferenceParams,
           const uint64 initialModelVersion)
    : m_clientArgs(clientArgs), m_args(mctsArgs), m_threadPool(mctsArgs.num_threads),
      m_arenaCapacity(mctsArgs.arenaCapacity()), m_modelVersion(initialModelVersion),
      m_directInferenceParams(std::move(directInferenceParams)) {
    if (m_directInferenceParams.has_value()) {
        if (useInferenceCache) {
            throw std::invalid_argument("Direct self-play inference does not support caching");
        }
        m_directSearch =
            std::make_unique<DirectSelfPlaySearch>(m_clientArgs, m_args, *m_directInferenceParams);
    } else if (useInferenceCache) {
        m_client = std::make_unique<InferenceClient>(m_clientArgs);
    } else {
        m_client = std::make_unique<NonCachingInferenceClient>(m_clientArgs);
    }
}

MCTS::~MCTS() = default;

uint32 MCTSParams::arenaCapacity() const {
    const uint64 maximumSearches = std::max(num_full_searches, num_fast_searches);
    const uint64 capacity = maximumSearches + static_cast<uint64>(num_parallel_searches) + 1U;
    if (capacity > std::numeric_limits<uint32>::max()) {
        throw std::overflow_error("MCTS search parameters exceed the node index capacity");
    }
    return static_cast<uint32>(capacity);
}

MCTSRoot MCTS::newRoot(const std::string &fen) const {
    const std::shared_lock operationLock(m_operationMutex);
    return MCTSRoot::create(fen, m_arenaCapacity);
}

MCTSRoot MCTS::newRoot(Board board) const {
    const std::shared_lock operationLock(m_operationMutex);
    return MCTSRoot::create(std::move(board), m_arenaCapacity);
}

uint32 MCTS::arenaCapacity() const {
    const std::shared_lock operationLock(m_operationMutex);
    return m_arenaCapacity;
}

uint64 MCTS::modelVersion() const {
    const std::shared_lock operationLock(m_operationMutex);
    return m_modelVersion;
}

MCTSResults MCTS::searchGames(const std::vector<MCTSBoard> &boards) {
    TIMEIT("MCTS::searchGames");

    const size_t numberOfBoards = boards.size();
    std::vector<MCTSRoot> roots;
    roots.reserve(numberOfBoards);

    std::vector<size_t> newBoardIndices;
    std::vector<const Board *> newBoards;
    newBoardIndices.reserve(numberOfBoards);
    newBoards.reserve(numberOfBoards);

    for (const auto &[index, board] : enumerate(boards)) {
        const MCTSRoot &root = board.root;
        if (root.arenaCapacity() != m_arenaCapacity) {
            throw std::invalid_argument("MCTSRoot arena capacity does not match MCTS parameters");
        }
        roots.push_back(root);

        if (!root.isExpanded()) {
            newBoardIndices.emplace_back(index);
            newBoards.emplace_back(&root.board());
        }
    }

    const std::vector<InferenceResult> inferenceResults = inferenceBatchUnlocked(newBoards);
    for (const auto [rootIndex, result] : zip(newBoardIndices, inferenceResults)) {
        roots[rootIndex].tree().expand(roots[rootIndex].rootIndex(), result.moves);
    }

    for (size_t index = 0; index < roots.size(); ++index) {
        if (boards[index].should_run_full_search) {
            addNoise(roots[index]);
        }
    }

    std::vector<RootTask> active;
    active.reserve(numberOfBoards);
    uint64 startingVisits = 0;
    for (size_t index = 0; index < numberOfBoards; ++index) {
        const uint32 limit = boards[index].should_run_full_search ? m_args.num_full_searches
                                                                  : m_args.num_fast_searches;
        roots[index].tree().prepareForSearch(limit,
                                             static_cast<uint32>(m_args.num_parallel_searches));
        active.push_back({roots[index], limit});
        startingVisits += roots[index].visits();
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
    const uint64 finalVisits = std::accumulate(
        roots.begin(), roots.end(), uint64{0},
        [](const uint64 total, const MCTSRoot &root) { return total + root.visits(); });
    assert(finalVisits >= startingVisits);
    return {.results = std::move(results),
            .mctsStats = {},
            .searchesCompleted = finalVisits - startingVisits};
}

MCTSResults MCTS::search(const std::vector<MCTSBoard> &boards, const bool collectStatistics) {
    TIMEIT("MCTS::search");
    const std::shared_lock operationLock(m_operationMutex);
    if (m_directSearch != nullptr) {
        return m_directSearch->search(boards, collectStatistics);
    }
    if (boards.empty()) {
        return {.results = {}, .mctsStats = {}, .searchesCompleted = 0};
    }

    const size_t numberOfBoards = boards.size();
    const size_t numberOfThreads = std::max<size_t>(1, m_threadPool.numThreads());
    const size_t sliceSize = (numberOfBoards + numberOfThreads - 1) / numberOfThreads;

    std::vector<std::future<MCTSResults>> futures;
    futures.reserve(numberOfThreads);
    for (size_t slice = 0; slice < numberOfThreads && slice * sliceSize < numberOfBoards; ++slice) {
        const auto begin = boards.begin() + static_cast<std::ptrdiff_t>(slice * sliceSize);
        const auto end = begin + static_cast<std::ptrdiff_t>(
                                     std::min(sliceSize, numberOfBoards - slice * sliceSize));
        std::vector<MCTSBoard> slicedBoards(begin, end);
        futures.emplace_back(
            m_threadPool.enqueue(&MCTS::searchGames, this, std::move(slicedBoards)));
    }

    std::vector<MCTSResult> results;
    results.reserve(numberOfBoards);
    uint64 searchesCompleted = 0;
    for (std::future<MCTSResults> &future : futures) {
        MCTSResults sliceResults = future.get();
        searchesCompleted += sliceResults.searchesCompleted;
        results.insert(results.end(), std::make_move_iterator(sliceResults.results.begin()),
                       std::make_move_iterator(sliceResults.results.end()));
    }

    const MCTSStatistics statistics =
        collectStatistics ? mctsStatistics(results.front().root) : MCTSStatistics{};
    return {.results = std::move(results),
            .mctsStats = statistics,
            .searchesCompleted = searchesCompleted};
}

std::pair<InferenceStatistics, TimeInfo> MCTS::getInferenceStatistics() {
    const std::shared_lock operationLock(m_operationMutex);
    const InferenceStatistics statistics =
        m_directSearch != nullptr
            ? m_directSearch->inferenceStatistics()
            : std::visit(
                  [](auto &client) -> InferenceStatistics {
                      using Client = std::decay_t<decltype(client)>;
                      if constexpr (std::same_as<Client, std::monostate>) {
                          throw std::logic_error("MCTS inference is not configured");
                      } else {
                          return client->getStatistics();
                      }
                  },
                  m_client);
    return {statistics, resetTimes()};
}

void MCTS::refreshModel(const uint64 modelVersion, const std::string &modelPath) {
    const std::unique_lock operationLock(m_operationMutex);
    if (modelVersion <= m_modelVersion) {
        throw std::invalid_argument("Refreshed model version must increase");
    }
    if (m_directSearch != nullptr) {
        m_directSearch->refreshModel(modelPath);
    } else {
        std::visit(
            [&modelPath](auto &client) {
                using Client = std::decay_t<decltype(client)>;
                if constexpr (!std::same_as<Client, std::monostate>) {
                    client->refreshModel(modelPath);
                }
            },
            m_client);
    }
    m_clientArgs.currentModelPath = modelPath;
    m_modelVersion = modelVersion;
}

bool MCTS::updateSearchSchedule(const MCTSParams &mctsArgs) {
    const std::unique_lock operationLock(m_operationMutex);
    if (mctsArgs.num_threads != m_args.num_threads) {
        throw std::invalid_argument("MCTS thread count cannot change during a persistent run");
    }
    const uint32 updatedArenaCapacity = mctsArgs.arenaCapacity();
    const bool arenaCapacityChanged = updatedArenaCapacity != m_arenaCapacity;
    m_args = mctsArgs;
    m_arenaCapacity = updatedArenaCapacity;
    if (m_directSearch != nullptr) {
        m_directSearch->updateSearchSchedule(mctsArgs);
    }
    return arenaCapacityChanged;
}

std::vector<InferenceResult> MCTS::inferenceBatch(const std::vector<const Board *> &boards) {
    const std::shared_lock operationLock(m_operationMutex);
    return inferenceBatchUnlocked(boards);
}

std::vector<InferenceResult>
MCTS::inferenceBatchUnlocked(const std::vector<const Board *> &boards) {
    if (m_directSearch != nullptr) {
        std::vector<InferenceResult> results;
        results.reserve(boards.size());
        for (const Board *board : boards) {
            results.push_back(m_directSearch->evaluate(*board));
        }
        return results;
    }
    return std::visit(
        [&boards](auto &client) -> std::vector<InferenceResult> {
            using Client = std::decay_t<decltype(client)>;
            if constexpr (std::same_as<Client, std::monostate>) {
                throw std::logic_error("MCTS inference is not configured");
            } else {
                return client->inferenceBatch(boards);
            }
        },
        m_client);
}

std::vector<std::uintptr_t> MCTS::directWorkerIdentityTokens() const {
    const std::shared_lock operationLock(m_operationMutex);
    if (m_directSearch == nullptr) {
        return {};
    }
    return m_directSearch->workerIdentityTokens();
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

    const std::vector<InferenceResult> results = inferenceBatchUnlocked(selectedBoards);
    for (const auto [selected, result] : zip(selectedNodes, results)) {
        selected.tree->expand(selected.index, result.moves);
        selected.tree->backPropagateAndRemoveVirtualLoss(selected.index, result.value());
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
