#include "MCTS.hpp"

#include "DirectSelfPlaySearch.hpp"

MCTSParams::MCTSParams(const int numParallelSearches, const uint32 numFullSearches,
                       const uint32 numFastSearches, const float cParam,
                       const float dirichletAlpha, const float dirichletEpsilon,
                       const uint8 minVisitCount, const uint8 numThreads)
    : num_parallel_searches(numParallelSearches), num_full_searches(numFullSearches),
      num_fast_searches(numFastSearches), c_param(cParam), dirichlet_alpha(dirichletAlpha),
      dirichlet_epsilon(dirichletEpsilon), min_visit_count(minVisitCount),
      num_threads(numThreads) {
    if (num_parallel_searches <= 0 || num_full_searches == 0 || num_fast_searches == 0 ||
        num_threads == 0) {
        throw std::invalid_argument("MCTS search counts and thread count must be positive");
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

DirectSelfPlayInferenceParams::DirectSelfPlayInferenceParams(
    const int inferenceWorkers, const int inferenceBatchSize,
    const int outstandingBatchesPerWorker)
    : inference_workers(inferenceWorkers), inference_batch_size(inferenceBatchSize),
      outstanding_batches_per_worker(outstandingBatchesPerWorker) {
    if (inference_workers <= 0 || inference_batch_size <= 0) {
        throw std::invalid_argument("Direct inference worker and batch counts must be positive");
    }
    if (outstanding_batches_per_worker <= 0 || outstanding_batches_per_worker > 2) {
        throw std::invalid_argument("outstanding_batches_per_worker must be 1 or 2");
    }
}

MCTS::MCTS(const InferenceClientParams &clientArgs, const MCTSParams &mctsArgs,
           DirectSelfPlayInferenceParams directInferenceParams,
           const uint64 initialModelVersion)
    : m_clientArgs(clientArgs), m_args(mctsArgs), m_arenaCapacity(mctsArgs.arenaCapacity()),
      m_modelVersion(initialModelVersion),
      m_directInferenceParams(std::move(directInferenceParams)),
      m_directSearch(std::make_unique<DirectSelfPlaySearch>(m_clientArgs, m_args,
                                                            m_directInferenceParams)) {}

MCTS::~MCTS() = default;

MCTSRoot MCTS::newRoot(const std::string &fen) const {
    const std::shared_lock lock(m_operationMutex);
    return MCTSRoot::create(fen, m_arenaCapacity);
}

MCTSRoot MCTS::newRoot(Board board) const {
    const std::shared_lock lock(m_operationMutex);
    return MCTSRoot::create(std::move(board), m_arenaCapacity);
}

uint32 MCTS::arenaCapacity() const {
    const std::shared_lock lock(m_operationMutex);
    return m_arenaCapacity;
}

uint64 MCTS::modelVersion() const {
    const std::shared_lock lock(m_operationMutex);
    return m_modelVersion;
}

MCTSResults MCTS::search(const std::vector<MCTSBoard> &boards, const bool collectStatistics) {
    const std::shared_lock lock(m_operationMutex);
    return m_directSearch->search(boards, collectStatistics);
}

std::pair<InferenceStatistics, TimeInfo> MCTS::getInferenceStatistics() {
    const std::shared_lock lock(m_operationMutex);
    return {m_directSearch->inferenceStatistics(), resetTimes()};
}

void MCTS::refreshModel(const uint64 modelVersion, const std::string &modelPath) {
    const std::unique_lock lock(m_operationMutex);
    if (modelVersion <= m_modelVersion) {
        throw std::invalid_argument("Refreshed model version must increase");
    }
    m_directSearch->refreshModel(modelPath);
    m_clientArgs.currentModelPath = modelPath;
    m_modelVersion = modelVersion;
}

bool MCTS::updateSearchSchedule(const MCTSParams &mctsArgs) {
    const std::unique_lock lock(m_operationMutex);
    if (mctsArgs.num_threads != m_args.num_threads) {
        throw std::invalid_argument("MCTS thread count cannot change during a persistent run");
    }
    const uint32 updatedCapacity = mctsArgs.arenaCapacity();
    const bool capacityChanged = updatedCapacity != m_arenaCapacity;
    m_args = mctsArgs;
    m_arenaCapacity = updatedCapacity;
    m_directSearch->updateSearchSchedule(mctsArgs);
    return capacityChanged;
}

std::vector<InferenceResult> MCTS::inferenceBatch(const std::vector<const Board *> &boards) {
    const std::shared_lock lock(m_operationMutex);
    return inferenceBatchUnlocked(boards);
}

std::vector<InferenceResult>
MCTS::inferenceBatchUnlocked(const std::vector<const Board *> &boards) {
    std::vector<InferenceResult> results;
    results.reserve(boards.size());
    for (const Board *board : boards) {
        results.push_back(m_directSearch->evaluate(*board));
    }
    return results;
}

std::vector<std::uintptr_t> MCTS::directWorkerIdentityTokens() const {
    const std::shared_lock lock(m_operationMutex);
    return m_directSearch->workerIdentityTokens();
}
