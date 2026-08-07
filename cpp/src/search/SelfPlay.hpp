#pragma once

#include "search/InferenceConfiguration.hpp"
#include "search/SearchEngine.hpp"
#include "util/TimeItGuard.h"
#include "util/py.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <ranges>
#include <shared_mutex>
#include <stdexcept>
#include <utility>
#include <vector>

// Applies training search schedules and root noise across games through the shared engine.

struct SelfPlaySearchParameters {
    std::uint32_t parallel_searches;
    std::uint32_t full_searches;
    std::uint32_t fast_searches;
    float exploration_constant;
    float dirichlet_alpha;
    float dirichlet_epsilon;
    std::uint32_t minimum_root_visits;

    SelfPlaySearchParameters(std::uint32_t parallelSearches, std::uint32_t fullSearches,
                             std::uint32_t fastSearches, float explorationConstant,
                             float dirichletAlpha, float dirichletEpsilon,
                             std::uint32_t minimumRootVisits)
        : parallel_searches(parallelSearches), full_searches(fullSearches),
          fast_searches(fastSearches), exploration_constant(explorationConstant),
          dirichlet_alpha(dirichletAlpha), dirichlet_epsilon(dirichletEpsilon),
          minimum_root_visits(minimumRootVisits) {
        if (parallel_searches == 0 || full_searches == 0 || fast_searches == 0) {
            throw std::invalid_argument("Self-play search counts must be positive");
        }
    }

    [[nodiscard]] std::uint32_t arenaCapacity() const {
        const std::uint64_t maximumSearches = std::max(full_searches, fast_searches);
        const std::uint64_t capacity = maximumSearches + parallel_searches + 1U;
        if (capacity > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Search parameters exceed the node index capacity");
        }
        return static_cast<std::uint32_t>(capacity);
    }
};

struct SelfPlaySearchStatistics {
    float average_depth = 0.0F;
    float average_entropy = 0.0F;
    float average_kl_divergence = 0.0F;
    float average_policy_search_kl_divergence = 0.0F;
    float top_action_disagreement = 0.0F;
    float selected_action_prior_rank = 0.0F;
};

template <typename Game> struct SelfPlaySearchRequest {
    GameSearchRoot<Game> root;
    bool full_search;
};

template <typename Game> struct SelfPlaySearchResult {
    float root_value;
    std::vector<GameSearchVisit> visits;
    GameSearchRoot<Game> root;
};

template <typename Game> struct SelfPlaySearchBatch {
    std::vector<SelfPlaySearchResult<Game>> results;
    SelfPlaySearchStatistics statistics;
    std::uint64_t simulations_completed;
};

template <typename Game> class GameSelfPlaySearch {
public:
    using Position = typename Game::Position;
    using Root = GameSearchRoot<Game>;
    using Request = SelfPlaySearchRequest<Game>;
    using Result = SelfPlaySearchResult<Game>;
    using Batch = SelfPlaySearchBatch<Game>;

    GameSelfPlaySearch(const InferenceConfiguration &runtimeParameters,
                       const SelfPlaySearchParameters &searchParameters,
                       BatchedInferenceParameters inferenceParameters,
                       const std::uint64_t initialModelGeneration = 0)
        : m_runtimeParameters(runtimeParameters), m_searchParameters(searchParameters),
          m_arenaCapacity(searchParameters.arenaCapacity()),
          m_modelGeneration(initialModelGeneration),
          m_inferenceParameters(std::move(inferenceParameters)),
          m_search(std::make_unique<BatchedGameSearch<Game>>(
              m_runtimeParameters.model_path, m_runtimeParameters.device,
              m_runtimeParameters.device_id, m_inferenceParameters,
              engineParameters(m_searchParameters), initialModelGeneration, true,
              Game::searchTurnDiscount())) {}

    [[nodiscard]] Root newRoot(Position position) const {
        const std::shared_lock lock(m_operationMutex);
        return m_search->newRoot(std::move(position));
    }

    [[nodiscard]] Batch search(const std::vector<Request> &requests,
                               const bool collectStatistics = false) {
        const std::shared_lock lock(m_operationMutex);
        if (requests.empty()) {
            return {
                .results = {},
                .statistics = {},
                .simulations_completed = 0,
            };
        }
        std::vector<GameSearchRequest<Game>> engineRequests;
        engineRequests.reserve(requests.size());
        for (const Request &request : requests) {
            if (request.root.tree().capacity() != m_arenaCapacity) {
                throw std::invalid_argument(
                    "Root arena capacity does not match self-play search parameters");
            }
            engineRequests.push_back({
                .root = request.root,
                .visit_limit = request.full_search ? m_searchParameters.full_searches
                                                   : m_searchParameters.fast_searches,
                .add_root_noise = request.full_search,
            });
        }
        GameSearchBatchResult searched = m_search->searchDetailed(engineRequests);
        std::vector<Result> results;
        results.reserve(requests.size());
        for (const auto index : range(requests.size())) {
            results.push_back({
                .root_value = searched.results[index].root_value,
                .visits = std::move(searched.results[index].visits),
                .root = requests[index].root,
            });
        }
        const SelfPlaySearchStatistics collected =
            collectStatistics ? treeStatistics(results.front().root) : SelfPlaySearchStatistics{};
        return {
            .results = std::move(results),
            .statistics = collected,
            .simulations_completed = searched.simulations_completed,
        };
    }

    [[nodiscard]] std::uint32_t arenaCapacity() const {
        const std::shared_lock lock(m_operationMutex);
        return m_arenaCapacity;
    }

    [[nodiscard]] std::uint64_t modelGeneration() const {
        const std::shared_lock lock(m_operationMutex);
        return m_modelGeneration;
    }

    [[nodiscard]] std::pair<InferenceStatistics, TimeInfo> inferenceStatistics() {
        const std::shared_lock lock(m_operationMutex);
        return {m_search->inferenceStatistics(), resetTimes()};
    }

    void refreshModel(const std::uint64_t modelGeneration, const std::string &modelPath) {
        const std::unique_lock lock(m_operationMutex);
        m_search->refreshModel(modelGeneration, modelPath);
        m_runtimeParameters.model_path = modelPath;
        m_modelGeneration = modelGeneration;
    }

    [[nodiscard]] bool updateSearchSchedule(const SelfPlaySearchParameters &parameters) {
        const std::unique_lock lock(m_operationMutex);
        const std::uint32_t updatedCapacity = parameters.arenaCapacity();
        const bool capacityChanged = updatedCapacity != m_arenaCapacity;
        m_searchParameters = parameters;
        m_arenaCapacity = updatedCapacity;
        m_search->updateSearchParameters(engineParameters(m_searchParameters));
        return capacityChanged;
    }

    [[nodiscard]] std::vector<SearchInferenceResult<Game>>
    evaluate(const std::vector<Position> &positions) {
        const std::shared_lock lock(m_operationMutex);
        return m_search->evaluate(positions);
    }

    [[nodiscard]] std::vector<std::uintptr_t> workerIdentityTokens() const {
        const std::shared_lock lock(m_operationMutex);
        return m_search->workerIdentityTokens();
    }

private:
    InferenceConfiguration m_runtimeParameters;
    SelfPlaySearchParameters m_searchParameters;
    std::uint32_t m_arenaCapacity;
    std::uint64_t m_modelGeneration;
    BatchedInferenceParameters m_inferenceParameters;
    std::unique_ptr<BatchedGameSearch<Game>> m_search;
    mutable std::shared_mutex m_operationMutex;

    [[nodiscard]] static BatchedSearchParameters
    engineParameters(const SelfPlaySearchParameters &parameters) {
        return {parameters.parallel_searches,   parameters.exploration_constant,
                parameters.minimum_root_visits, parameters.dirichlet_alpha,
                parameters.dirichlet_epsilon,   parameters.arenaCapacity()};
    }

    [[nodiscard]] static SelfPlaySearchStatistics treeStatistics(const Root &root) {
        SelfPlaySearchStatistics result;
        result.average_depth = static_cast<float>(root.tree().maximumDepth());
        const GameSearchNode<Game> &rootNode = root.tree().root();
        if (rootNode.children.empty()) {
            return result;
        }
        const float totalVisits = static_cast<float>(root.visits());
        const float uniformProbability = 1.0F / static_cast<float>(rootNode.children.size());
        const GameSearchEdge<typename Game::Action> *mostVisited = nullptr;
        const GameSearchEdge<typename Game::Action> *highestPrior = nullptr;
        for (const GameSearchEdge<typename Game::Action> &child : rootNode.children) {
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
            result.average_kl_divergence +=
                probability * std::log2(probability / uniformProbability);
            result.average_policy_search_kl_divergence +=
                probability * std::log2(probability / std::max(child.raw_prior, 1e-12F));
        }
        if (mostVisited != nullptr) {
            result.top_action_disagreement =
                mostVisited->action == highestPrior->action ? 0.0F : 1.0F;
            result.selected_action_prior_rank =
                1.0F + static_cast<float>(std::ranges::count_if(
                           rootNode.children,
                           [mostVisited](const GameSearchEdge<typename Game::Action> &candidate) {
                               return candidate.raw_prior > mostVisited->raw_prior;
                           }));
        }
        return result;
    }
};
