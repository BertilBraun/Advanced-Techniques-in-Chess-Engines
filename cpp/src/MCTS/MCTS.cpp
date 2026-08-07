#include "MCTS.hpp"

#include <numeric>

namespace {
BatchedSearchParameters searchParameters(const MCTSParams &parameters) {
    return BatchedSearchParameters(
        static_cast<std::uint32_t>(parameters.num_parallel_searches), parameters.c_param,
        parameters.min_visit_count, parameters.dirichlet_alpha, parameters.dirichlet_epsilon,
        parameters.arenaCapacity());
}

MCTSStatistics statistics(const MCTSRoot &root) {
    MCTSStatistics result;
    result.averageDepth = static_cast<float>(root.maxDepth());
    const ChessSearchNode &rootNode = root.tree().root();
    if (rootNode.children.empty()) {
        return result;
    }
    const float totalVisits = static_cast<float>(root.visits());
    const float uniformProbability = 1.0F / static_cast<float>(rootNode.children.size());
    const ChessSearchEdge *mostVisited = nullptr;
    const ChessSearchEdge *highestPrior = nullptr;
    for (const ChessSearchEdge &child : rootNode.children) {
        if (mostVisited == nullptr || child.visits > mostVisited->visits) {
            mostVisited = &child;
        }
        if (highestPrior == nullptr || child.raw_prior > highestPrior->raw_prior) {
            highestPrior = &child;
        }
        if (child.visits == 0) {
            continue;
        }
        const float probability = static_cast<float>(child.visits) / totalVisits;
        result.averageEntropy -= probability * std::log2(probability);
        result.averageKLDivergence += probability * std::log2(probability / uniformProbability);
        result.averagePolicySearchKLDivergence +=
            probability * std::log2(probability / std::max(child.raw_prior, 1e-12F));
    }
    if (mostVisited != nullptr) {
        result.topMoveDisagreement = mostVisited->action == highestPrior->action ? 0.0F : 1.0F;
        result.selectedMovePriorRank =
            1.0F + static_cast<float>(std::ranges::count_if(
                       rootNode.children, [mostVisited](const ChessSearchEdge &candidate) {
                           return candidate.raw_prior > mostVisited->raw_prior;
                       }));
    }
    return result;
}
} // namespace

MCTSParams::MCTSParams(const int numParallelSearches, const std::uint32_t numFullSearches,
                       const std::uint32_t numFastSearches, const float cParam,
                       const float dirichletAlpha, const float dirichletEpsilon,
                       const std::uint8_t minVisitCount, const std::uint8_t numThreads)
    : num_parallel_searches(numParallelSearches), num_full_searches(numFullSearches),
      num_fast_searches(numFastSearches), c_param(cParam), dirichlet_alpha(dirichletAlpha),
      dirichlet_epsilon(dirichletEpsilon), min_visit_count(minVisitCount),
      num_threads(numThreads) {
    if (num_parallel_searches <= 0 || num_full_searches == 0 || num_fast_searches == 0 ||
        num_threads == 0) {
        throw std::invalid_argument("MCTS search counts and thread count must be positive");
    }
}

std::uint32_t MCTSParams::arenaCapacity() const {
    const std::uint64_t maximumSearches = std::max(num_full_searches, num_fast_searches);
    const std::uint64_t capacity = maximumSearches +
                                   static_cast<std::uint64_t>(num_parallel_searches) + 1U;
    if (capacity > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("MCTS search parameters exceed the node index capacity");
    }
    return static_cast<std::uint32_t>(capacity);
}

MCTS::MCTS(const InferenceClientParams &clientArgs, const MCTSParams &mctsArgs,
           BatchedInferenceParameters inferenceParameters,
           const std::uint64_t initialModelVersion)
    : m_clientArgs(clientArgs), m_args(mctsArgs), m_arenaCapacity(mctsArgs.arenaCapacity()),
      m_modelVersion(initialModelVersion),
      m_inferenceParameters(std::move(inferenceParameters)),
      m_search(std::make_unique<BatchedGameSearch<ChessGameContract>>(
          m_clientArgs.currentModelPath, m_clientArgs.device, m_clientArgs.device_id,
          m_inferenceParameters, searchParameters(m_args),
          initialModelVersion, false, 0.99F)) {}

MCTS::~MCTS() = default;

MCTSRoot MCTS::newRoot(const std::string &fen) const { return newRoot(Board(fen)); }

MCTSRoot MCTS::newRoot(Board board) const {
    const std::shared_lock lock(m_operationMutex);
    return MCTSRoot(m_search->newRoot(std::move(board)));
}

std::uint32_t MCTS::arenaCapacity() const {
    const std::shared_lock lock(m_operationMutex);
    return m_arenaCapacity;
}

std::uint64_t MCTS::modelVersion() const {
    const std::shared_lock lock(m_operationMutex);
    return m_modelVersion;
}

MCTSResults MCTS::search(const std::vector<MCTSBoard> &boards, const bool collectStatistics) {
    const std::shared_lock lock(m_operationMutex);
    if (boards.empty()) {
        return {{}, {}, 0};
    }
    std::vector<GameSearchRequest<ChessGameContract>> requests;
    requests.reserve(boards.size());
    for (const MCTSBoard &board : boards) {
        if (board.root.arenaCapacity() != m_arenaCapacity) {
            throw std::invalid_argument("MCTS root arena capacity does not match search parameters");
        }
        requests.push_back(
            {board.root.gameRoot(),
             board.should_run_full_search ? m_args.num_full_searches : m_args.num_fast_searches,
             board.should_run_full_search});
    }
    GameSearchBatchResult searched = m_search->searchDetailed(requests);
    std::vector<MCTSResult> results;
    results.reserve(boards.size());
    for (std::size_t index = 0; index < boards.size(); ++index) {
        VisitCounts visits;
        visits.reserve(searched.results[index].visits.size());
        for (const GameSearchVisit &visit : searched.results[index].visits) {
            visits.emplace_back(visit.action_id, static_cast<int>(visit.visit_count));
        }
        results.push_back({searched.results[index].root_value, std::move(visits),
                           boards[index].root});
    }
    const MCTSStatistics collected =
        collectStatistics ? statistics(results.front().root) : MCTSStatistics{};
    return {std::move(results), collected, searched.simulations_completed};
}

std::pair<InferenceStatistics, TimeInfo> MCTS::getInferenceStatistics() {
    const std::shared_lock lock(m_operationMutex);
    return {m_search->inferenceStatistics(), resetTimes()};
}

void MCTS::refreshModel(const std::uint64_t modelVersion, const std::string &modelPath) {
    const std::unique_lock lock(m_operationMutex);
    m_search->refreshModel(modelVersion, modelPath);
    m_clientArgs.currentModelPath = modelPath;
    m_modelVersion = modelVersion;
}

bool MCTS::updateSearchSchedule(const MCTSParams &mctsArgs) {
    const std::unique_lock lock(m_operationMutex);
    if (mctsArgs.num_threads != m_args.num_threads) {
        throw std::invalid_argument("MCTS thread count cannot change during a persistent run");
    }
    const std::uint32_t updatedCapacity = mctsArgs.arenaCapacity();
    const bool capacityChanged = updatedCapacity != m_arenaCapacity;
    m_args = mctsArgs;
    m_arenaCapacity = updatedCapacity;
    m_search->updateSearchParameters(searchParameters(m_args));
    return capacityChanged;
}

std::vector<InferenceResult> MCTS::inferenceBatch(const std::vector<const Board *> &boards) {
    const std::shared_lock lock(m_operationMutex);
    return inferenceBatchUnlocked(boards);
}

std::vector<InferenceResult>
MCTS::inferenceBatchUnlocked(const std::vector<const Board *> &boards) {
    std::vector<Board> positions;
    positions.reserve(boards.size());
    for (const Board *board : boards) {
        positions.push_back(*board);
    }
    return m_search->evaluate(positions);
}

std::vector<std::uintptr_t> MCTS::directWorkerIdentityTokens() const {
    const std::shared_lock lock(m_operationMutex);
    return m_search->workerIdentityTokens();
}
