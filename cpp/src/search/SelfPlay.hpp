#pragma once

#include "games/GameConcepts.hpp"
#include "search/InferenceConfiguration.hpp"
#include "search/SearchEngine.hpp"
#include "util/py.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <shared_mutex>
#include <stdexcept>
#include <utility>
#include <vector>

struct SelfPlaySearchParameters {
    std::uint32_t baseline_visits;
    SearchBudgetPolicy search_budget_policy;
    TreeSearchParameters tree_search;
    float dirichlet_alpha;
    float dirichlet_epsilon;

    SelfPlaySearchParameters(const std::uint32_t baselineVisits,
                             SearchBudgetPolicy searchBudgetPolicy, TreeSearchParameters treeSearch,
                             const float dirichletAlpha, const float dirichletEpsilon)
        : baseline_visits(baselineVisits), search_budget_policy(std::move(searchBudgetPolicy)),
          tree_search(treeSearch), dirichlet_alpha(dirichletAlpha),
          dirichlet_epsilon(dirichletEpsilon) {
        static_cast<void>(PredictedSearchBudgetLimit(baseline_visits, search_budget_policy));
    }

    [[nodiscard]] std::uint32_t initialArenaCapacity() const {
        return checkedArenaCapacity(baseline_visits, searchParallelism(baseline_visits));
    }

    [[nodiscard]] std::uint32_t maximumArenaCapacity() const {
        const std::uint64_t maximumSearches = maximumAdditionalVisits(
            PredictedSearchBudgetLimit(baseline_visits, search_budget_policy));
        return checkedArenaCapacity(maximumSearches, 16U);
    }

private:
    [[nodiscard]] static std::uint32_t checkedArenaCapacity(const std::uint64_t visits,
                                                            const std::uint64_t parallelSearches) {
        const std::uint64_t capacity = visits + parallelSearches + 1U;
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

template <SearchGame Game> struct SelfPlaySearchRequest {
    GameSearchRoot<Game> root;
    std::optional<std::uint32_t> assigned_additional_visits;
    std::vector<std::uint32_t> policy_checkpoint_visits;
    std::optional<std::uint32_t> parallel_searches;
    bool add_root_noise;
    bool force_root_playouts;
    SearchCheckpointDetail checkpoint_detail = SearchCheckpointDetail::Scalars;
};

template <SearchGame Game> struct SelfPlaySearchResult {
    float root_value;
    int highest_visited_child_action_id;
    std::uint32_t highest_visited_child_visit_count;
    float highest_visited_child_q;
    std::vector<GameSearchVisit> search_visits;
    std::vector<GameSearchVisit> policy_target_visits;
    float network_root_value;
    float policy_correction;
    float value_correction;
    SearchBudgetCurvePrediction predicted_budget_curve;
    int selected_budget_index;
    std::uint32_t assigned_additional_visits;
    std::uint32_t parallel_searches;
    std::int64_t spend_residual;
    std::uint32_t starting_visits;
    std::uint32_t final_visits;
    SearchStopReason stop_reason;
    std::vector<SearchCheckpoint> checkpoints;
    GameSearchRoot<Game> root;
};

template <SearchGame Game> struct SelfPlaySearchBatch {
    std::vector<SelfPlaySearchResult<Game>> results;
    SelfPlaySearchStatistics statistics;
    std::uint64_t simulations_completed;
};

template <SearchGame Game> class GameSelfPlaySearch {
public:
    using Position = typename Game::State;
    using Root = GameSearchRoot<Game>;
    using Request = SelfPlaySearchRequest<Game>;
    using Result = SelfPlaySearchResult<Game>;
    using Batch = SelfPlaySearchBatch<Game>;

    GameSelfPlaySearch(const InferenceConfiguration &runtimeParameters,
                       const SelfPlaySearchParameters &searchParameters,
                       BatchedInferenceParameters inferenceParameters,
                       const std::uint64_t initialModelGeneration = 0)
        : m_runtimeParameters(runtimeParameters), m_searchParameters(searchParameters),
          m_arenaCapacity(searchParameters.maximumArenaCapacity()),
          m_inferenceParameters(std::move(inferenceParameters)),
          m_search(std::make_unique<BatchedGameSearch<Game>>(
              m_runtimeParameters.model_path, m_runtimeParameters.device,
              m_runtimeParameters.device_id, m_inferenceParameters,
              engineParameters(m_searchParameters), initialModelGeneration, true,
              m_runtimeParameters.execution_options)) {}

    [[nodiscard]] Root newRoot(Position position, const std::size_t maximumCapacity = 0) const {
        const std::shared_lock lock(m_operationMutex);
        return m_search->newRoot(std::move(position), maximumCapacity);
    }

    [[nodiscard]] Batch search(const std::vector<Request> &requests,
                               const bool collectStatistics = false) {
        const std::shared_lock lock(m_operationMutex);
        if (requests.empty()) {
            return {.results = {}, .statistics = {}, .simulations_completed = 0};
        }
        std::vector<GameSearchRequest<Game>> engineRequests;
        engineRequests.reserve(requests.size());
        for (const Request &request : requests) {
            const SearchLimit limit =
                request.assigned_additional_visits.has_value()
                    ? SearchLimit{AdditionalSearchLimit(*request.assigned_additional_visits)}
                    : SearchLimit{
                          PredictedSearchBudgetLimit(m_searchParameters.baseline_visits,
                                                     m_searchParameters.search_budget_policy)};
            engineRequests.push_back({
                .root = request.root,
                .limit = limit,
                .add_root_noise = request.add_root_noise,
                .force_root_playouts = request.force_root_playouts,
                .checkpoint_detail = request.checkpoint_detail,
                .policy_checkpoint_visits = request.policy_checkpoint_visits,
                .parallel_searches = request.parallel_searches,
            });
        }
        GameSearchBatchResult searched =
            m_search->searchDetailed(engineRequests, &m_budgetAllocator);
        std::vector<Result> results;
        results.reserve(requests.size());
        for (const auto index : range(requests.size())) {
            results.push_back({
                .root_value = searched.results[index].root_value,
                .highest_visited_child_action_id =
                    searched.results[index].highest_visited_child_action_id,
                .highest_visited_child_visit_count =
                    searched.results[index].highest_visited_child_visit_count,
                .highest_visited_child_q = searched.results[index].highest_visited_child_q,
                .search_visits = std::move(searched.results[index].search_visits),
                .policy_target_visits = std::move(searched.results[index].policy_target_visits),
                .network_root_value = searched.results[index].network_root_value,
                .policy_correction = searched.results[index].policy_correction,
                .value_correction = searched.results[index].value_correction,
                .predicted_budget_curve = searched.results[index].predicted_budget_curve,
                .selected_budget_index = searched.results[index].selected_budget_index,
                .assigned_additional_visits = searched.results[index].assigned_additional_visits,
                .parallel_searches = searched.results[index].parallel_searches,
                .spend_residual = searched.results[index].spend_residual,
                .starting_visits = searched.results[index].starting_visits,
                .final_visits = searched.results[index].final_visits,
                .stop_reason = searched.results[index].stop_reason,
                .checkpoints = std::move(searched.results[index].checkpoints),
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

    [[nodiscard]] std::int64_t spendResidual() const {
        const std::shared_lock lock(m_operationMutex);
        return m_budgetAllocator.spendError();
    }

    void resetSpendResidual() {
        const std::unique_lock lock(m_operationMutex);
        m_budgetAllocator.reset();
    }

    [[nodiscard]] std::uint64_t modelGeneration() const {
        const std::shared_lock lock(m_operationMutex);
        return m_search->modelGeneration();
    }

    [[nodiscard]] InferenceStatistics inferenceStatistics() const {
        const std::shared_lock lock(m_operationMutex);
        return m_search->inferenceStatistics();
    }

    void refreshModel(const std::uint64_t modelGeneration, const std::string &modelPath) {
        const std::unique_lock lock(m_operationMutex);
        m_search->refreshModel(modelGeneration, modelPath);
        m_runtimeParameters.model_path = modelPath;
    }

    [[nodiscard]] bool updateSearchSchedule(const SelfPlaySearchParameters &parameters) {
        const std::unique_lock lock(m_operationMutex);
        const std::uint32_t updatedCapacity = parameters.maximumArenaCapacity();
        const bool capacityChanged = updatedCapacity != m_arenaCapacity;
        const bool valueDiscountChanged = parameters.tree_search.value_discount_per_ply !=
                                          m_searchParameters.tree_search.value_discount_per_ply;
        m_searchParameters = parameters;
        m_arenaCapacity = updatedCapacity;
        m_search->updateSearchParameters(engineParameters(m_searchParameters));
        return capacityChanged || valueDiscountChanged;
    }

    [[nodiscard]] std::vector<SearchInferenceResult<Game>>
    evaluate(const std::vector<Position> &positions) {
        const std::shared_lock lock(m_operationMutex);
        return m_search->evaluate(positions);
    }

private:
    InferenceConfiguration m_runtimeParameters;
    SelfPlaySearchParameters m_searchParameters;
    std::uint32_t m_arenaCapacity;
    BatchedInferenceParameters m_inferenceParameters;
    std::unique_ptr<BatchedGameSearch<Game>> m_search;
    SearchBudgetAllocator m_budgetAllocator;
    mutable std::shared_mutex m_operationMutex;

    [[nodiscard]] static BatchedSearchParameters
    engineParameters(const SelfPlaySearchParameters &parameters) {
        return {parameters.tree_search, parameters.dirichlet_alpha, parameters.dirichlet_epsilon,
                parameters.initialArenaCapacity(), parameters.maximumArenaCapacity()};
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
