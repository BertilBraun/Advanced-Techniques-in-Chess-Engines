#include "games/chess/ChessSelfPlaySearch.hpp"

namespace {
BatchedSearchParameters
sharedSearchParameters(const ChessSelfPlaySearchParameters &parameters) {
    return BatchedSearchParameters(
        static_cast<std::uint32_t>(parameters.parallel_searches),
        parameters.exploration_constant, parameters.minimum_root_visits,
        parameters.dirichlet_alpha, parameters.dirichlet_epsilon, parameters.arenaCapacity());
}

ChessSelfPlaySearchStatistics statistics(const ChessSearchRoot &root) {
    ChessSelfPlaySearchStatistics result;
    result.average_depth = static_cast<float>(root.maxDepth());
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
        result.average_entropy -= probability * std::log2(probability);
        result.average_kl_divergence += probability * std::log2(probability / uniformProbability);
        result.average_policy_search_kl_divergence +=
            probability * std::log2(probability / std::max(child.raw_prior, 1e-12F));
    }
    if (mostVisited != nullptr) {
        result.top_action_disagreement =
            mostVisited->action == highestPrior->action ? 0.0F : 1.0F;
        result.selected_action_prior_rank =
            1.0F + static_cast<float>(std::ranges::count_if(
                       rootNode.children, [mostVisited](const ChessSearchEdge &candidate) {
                           return candidate.raw_prior > mostVisited->raw_prior;
                       }));
    }
    return result;
}
} // namespace

ChessSelfPlaySearchParameters::ChessSelfPlaySearchParameters(
    const int parallelSearches, const std::uint32_t fullSearches,
    const std::uint32_t fastSearches, const float explorationConstant,
    const float dirichletAlpha, const float dirichletEpsilon,
    const std::uint8_t minimumRootVisits)
    : parallel_searches(parallelSearches), full_searches(fullSearches),
      fast_searches(fastSearches), exploration_constant(explorationConstant),
      dirichlet_alpha(dirichletAlpha), dirichlet_epsilon(dirichletEpsilon),
      minimum_root_visits(minimumRootVisits) {
    if (parallel_searches <= 0 || full_searches == 0 || fast_searches == 0) {
        throw std::invalid_argument("Chess self-play search counts must be positive");
    }
}

std::uint32_t ChessSelfPlaySearchParameters::arenaCapacity() const {
    const std::uint64_t maximumSearches = std::max(full_searches, fast_searches);
    const std::uint64_t capacity =
        maximumSearches + static_cast<std::uint64_t>(parallel_searches) + 1U;
    if (capacity > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Chess search parameters exceed the node index capacity");
    }
    return static_cast<std::uint32_t>(capacity);
}

ChessSelfPlaySearch::ChessSelfPlaySearch(
    const InferenceClientParams &runtimeParameters,
    const ChessSelfPlaySearchParameters &searchParameters,
    BatchedInferenceParameters inferenceParameters, const std::uint64_t initialModelVersion)
    : m_runtimeParameters(runtimeParameters), m_searchParameters(searchParameters),
      m_arenaCapacity(searchParameters.arenaCapacity()), m_modelVersion(initialModelVersion),
      m_inferenceParameters(std::move(inferenceParameters)),
      m_search(std::make_unique<BatchedGameSearch<ChessGameContract>>(
          m_runtimeParameters.currentModelPath, m_runtimeParameters.device,
          m_runtimeParameters.device_id, m_inferenceParameters,
          sharedSearchParameters(m_searchParameters), initialModelVersion, false, 0.99F)) {}

ChessSelfPlaySearch::~ChessSelfPlaySearch() = default;

ChessSearchRoot ChessSelfPlaySearch::newRoot(const std::string &fen) const {
    return newRoot(Board(fen));
}

ChessSearchRoot ChessSelfPlaySearch::newRoot(Board board) const {
    const std::shared_lock lock(m_operationMutex);
    return ChessSearchRoot(m_search->newRoot(std::move(board)));
}

std::uint32_t ChessSelfPlaySearch::arenaCapacity() const {
    const std::shared_lock lock(m_operationMutex);
    return m_arenaCapacity;
}

std::uint64_t ChessSelfPlaySearch::modelVersion() const {
    const std::shared_lock lock(m_operationMutex);
    return m_modelVersion;
}

ChessSelfPlaySearchBatch
ChessSelfPlaySearch::search(const std::vector<ChessSelfPlaySearchRequest> &requests,
                            const bool collectStatistics) {
    const std::shared_lock lock(m_operationMutex);
    if (requests.empty()) {
        return {{}, {}, 0};
    }
    std::vector<GameSearchRequest<ChessGameContract>> sharedRequests;
    sharedRequests.reserve(requests.size());
    for (const ChessSelfPlaySearchRequest &request : requests) {
        if (request.root.arenaCapacity() != m_arenaCapacity) {
            throw std::invalid_argument("Chess root arena capacity does not match search parameters");
        }
        sharedRequests.push_back(
            {request.root.gameRoot(),
             request.full_search ? m_searchParameters.full_searches
                                 : m_searchParameters.fast_searches,
             request.full_search});
    }
    GameSearchBatchResult searched = m_search->searchDetailed(sharedRequests);
    std::vector<ChessSelfPlaySearchResult> results;
    results.reserve(requests.size());
    for (std::size_t index = 0; index < requests.size(); ++index) {
        ChessVisitCounts visits;
        visits.reserve(searched.results[index].visits.size());
        for (const GameSearchVisit &visit : searched.results[index].visits) {
            visits.emplace_back(visit.action_id, static_cast<int>(visit.visit_count));
        }
        results.push_back({searched.results[index].root_value, std::move(visits),
                           requests[index].root});
    }
    const ChessSelfPlaySearchStatistics collected =
        collectStatistics ? statistics(results.front().root) : ChessSelfPlaySearchStatistics{};
    return {std::move(results), collected, searched.simulations_completed};
}

std::pair<InferenceStatistics, TimeInfo> ChessSelfPlaySearch::inferenceStatistics() {
    const std::shared_lock lock(m_operationMutex);
    return {m_search->inferenceStatistics(), resetTimes()};
}

void ChessSelfPlaySearch::refreshModel(const std::uint64_t modelVersion,
                                       const std::string &modelPath) {
    const std::unique_lock lock(m_operationMutex);
    m_search->refreshModel(modelVersion, modelPath);
    m_runtimeParameters.currentModelPath = modelPath;
    m_modelVersion = modelVersion;
}

bool ChessSelfPlaySearch::updateSearchSchedule(
    const ChessSelfPlaySearchParameters &searchParameters) {
    const std::unique_lock lock(m_operationMutex);
    const std::uint32_t updatedCapacity = searchParameters.arenaCapacity();
    const bool capacityChanged = updatedCapacity != m_arenaCapacity;
    m_searchParameters = searchParameters;
    m_arenaCapacity = updatedCapacity;
    m_search->updateSearchParameters(sharedSearchParameters(m_searchParameters));
    return capacityChanged;
}

std::vector<InferenceResult>
ChessSelfPlaySearch::evaluate(const std::vector<const Board *> &boards) {
    const std::shared_lock lock(m_operationMutex);
    std::vector<Board> positions;
    positions.reserve(boards.size());
    for (const Board *board : boards) {
        positions.push_back(*board);
    }
    return m_search->evaluate(positions);
}

std::vector<std::uintptr_t> ChessSelfPlaySearch::workerIdentityTokens() const {
    const std::shared_lock lock(m_operationMutex);
    return m_search->workerIdentityTokens();
}
