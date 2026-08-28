#pragma once

#include "games/GameConcepts.hpp"
#include "search/InferenceTypes.hpp"
#include "search/SearchTree.hpp"
#include "search/tree/TreeSearchParameters.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

struct GameSearchVisit {
    int action_id;
    std::uint32_t visit_count;

    bool operator==(const GameSearchVisit &) const = default;
};

enum class SearchStopReason { FixedLimit, AdditionalVisits, PredictedBudget };
enum class SearchCheckpointDetail { Scalars, Policies };

struct SearchCheckpoint {
    std::uint32_t visits;
    std::vector<GameSearchVisit> policy_target_visits;
};

struct FixedSearchLimit {
    std::uint32_t visits;

    FixedSearchLimit() : visits(1) {}
    explicit FixedSearchLimit(const std::uint32_t visitLimit) : visits(visitLimit) {
        if (visits == 0) {
            throw std::invalid_argument("Fixed search limit must be positive");
        }
    }
};

struct AdditionalSearchLimit {
    std::uint32_t additional_visits;

    AdditionalSearchLimit() : additional_visits(1) {}
    explicit AdditionalSearchLimit(const std::uint32_t additionalVisits)
        : additional_visits(additionalVisits) {
        if (additional_visits == 0) {
            throw std::invalid_argument("Additional search limit must be positive");
        }
    }
};

struct SearchBudgetCurve {
    static constexpr std::size_t BUCKET_COUNT = 10;
    std::array<double, BUCKET_COUNT> multipliers;

    SearchBudgetCurve() : multipliers{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0} {}
    explicit SearchBudgetCurve(std::array<double, BUCKET_COUNT> curveMultipliers)
        : multipliers(std::move(curveMultipliers)) {
        if (std::ranges::any_of(multipliers, [](const double multiplier) {
                return !std::isfinite(multiplier) || multiplier <= 0.0;
            })) {
            throw std::invalid_argument(
                "Search-budget curve multipliers must be finite and positive");
        }
        if (!std::ranges::is_sorted(multipliers)) {
            throw std::invalid_argument(
                "Search-budget curve multipliers must be monotone nondecreasing");
        }
        const double sum = std::accumulate(multipliers.begin(), multipliers.end(), 0.0);
        if (std::abs(sum - static_cast<double>(BUCKET_COUNT)) > 1e-6) {
            throw std::invalid_argument(
                "Search-budget curve multipliers must have arithmetic mean one");
        }
    }

    [[nodiscard]] double multiplier(const float quantile) const {
        if (!std::isfinite(quantile) || quantile < 0.0F || quantile > 1.0F) {
            throw std::invalid_argument("Search-budget prediction must lie in [0, 1]");
        }
        const std::size_t bucket =
            std::min(static_cast<std::size_t>(quantile * static_cast<float>(BUCKET_COUNT)),
                     BUCKET_COUNT - 1);
        return multipliers[bucket];
    }
};

struct PredictedSearchBudgetLimit {
    std::uint32_t baseline_visits;
    SearchBudgetCurve curve;

    PredictedSearchBudgetLimit() : baseline_visits(1), curve() {}
    PredictedSearchBudgetLimit(const std::uint32_t baselineVisits, SearchBudgetCurve budgetCurve)
        : baseline_visits(baselineVisits), curve(std::move(budgetCurve)) {
        if (baseline_visits == 0) {
            throw std::invalid_argument("Predicted search-budget baseline must be positive");
        }
    }
};

using SearchLimit =
    std::variant<FixedSearchLimit, AdditionalSearchLimit, PredictedSearchBudgetLimit>;

[[nodiscard]] inline double searchBudgetMultiplier(const SearchBudgetCurve &curve,
                                                   const float quantile) {
    return curve.multiplier(quantile);
}

[[nodiscard]] inline std::uint32_t searchParallelism(const std::uint32_t additionalVisits) {
    if (additionalVisits == 0) {
        throw std::invalid_argument("Assigned additional visits must be positive");
    }
    const std::uint32_t targetRounds = (additionalVisits + 199U) / 200U;
    std::uint32_t parallelSearches = 1;
    while (parallelSearches < targetRounds && parallelSearches < 16U) {
        parallelSearches *= 2U;
    }
    return std::min(parallelSearches, 16U);
}

class SearchBudgetAllocator {
public:
    [[nodiscard]] std::uint32_t assign(const PredictedSearchBudgetLimit &limit,
                                       const float predictedQuantile) {
        const double multiplier = searchBudgetMultiplier(limit.curve, predictedQuantile);
        constexpr std::uint64_t maximumBaselineMultiple = 8;
        const std::uint64_t maximumVisits =
            static_cast<std::uint64_t>(limit.baseline_visits) * maximumBaselineMultiple;
        if (maximumVisits > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Predicted search budget exceeds the visit range");
        }
        const double ideal = static_cast<double>(limit.baseline_visits) * multiplier;
        const double corrected = ideal - static_cast<double>(m_spendError);
        const double rounded =
            std::clamp(std::floor(corrected + 0.5), 1.0, static_cast<double>(maximumVisits));
        if (rounded < 1.0 || rounded > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Predicted search budget is outside the visit range");
        }
        const auto assigned = static_cast<std::uint32_t>(rounded);
        const std::int64_t delta =
            static_cast<std::int64_t>(assigned) - static_cast<std::int64_t>(limit.baseline_visits);
        if ((delta > 0 && m_spendError > std::numeric_limits<std::int64_t>::max() - delta) ||
            (delta < 0 && m_spendError < std::numeric_limits<std::int64_t>::min() - delta)) {
            throw std::overflow_error("Search-budget spend ledger overflowed");
        }
        m_spendError += delta;
        return assigned;
    }

    [[nodiscard]] std::int64_t spendError() const noexcept { return m_spendError; }
    void reset() noexcept { m_spendError = 0; }

private:
    std::int64_t m_spendError = 0;
};

[[nodiscard]] inline std::uint32_t maximumAdditionalVisits(const SearchLimit &limit) {
    return std::visit(
        [](const auto &selected) -> std::uint32_t {
            using Limit = std::decay_t<decltype(selected)>;
            if constexpr (std::is_same_v<Limit, FixedSearchLimit>) {
                return selected.visits;
            } else if constexpr (std::is_same_v<Limit, AdditionalSearchLimit>) {
                return selected.additional_visits;
            } else {
                constexpr std::uint64_t maximumBaselineMultiple = 8;
                const std::uint64_t maximum =
                    static_cast<std::uint64_t>(selected.baseline_visits) * maximumBaselineMultiple;
                if (maximum > std::numeric_limits<std::uint32_t>::max()) {
                    throw std::overflow_error("Predicted search budget exceeds the visit range");
                }
                return static_cast<std::uint32_t>(maximum);
            }
        },
        limit);
}

struct GameSearchResult {
    float root_value;
    int highest_visited_child_action_id;
    std::uint32_t highest_visited_child_visit_count;
    float highest_visited_child_q;
    std::vector<GameSearchVisit> search_visits;
    std::vector<GameSearchVisit> policy_target_visits;
    float network_root_value;
    float policy_correction;
    float value_correction;
    float search_budget_logit;
    float predicted_search_budget;
    std::uint32_t assigned_additional_visits;
    std::uint32_t parallel_searches;
    std::int64_t spend_residual;
    std::uint32_t starting_visits;
    std::uint32_t final_visits;
    SearchStopReason stop_reason;
    std::vector<SearchCheckpoint> checkpoints;
};

template <SearchGame Game> struct GameSearchRequest {
    GameSearchRoot<Game> root;
    SearchLimit limit;
    bool add_root_noise;
    bool force_root_playouts = false;
    bool count_root_initialization = false;
    SearchCheckpointDetail checkpoint_detail = SearchCheckpointDetail::Scalars;
    std::vector<std::uint32_t> policy_checkpoint_visits;
    std::optional<std::uint32_t> parallel_searches;
};

struct GameSearchBatchResult {
    std::vector<GameSearchResult> results;
    std::uint64_t simulations_completed;
};

struct BatchedSearchParameters {
    TreeSearchParameters tree_search;
    float dirichlet_alpha;
    float dirichlet_epsilon;
    std::size_t tree_capacity;

    BatchedSearchParameters(TreeSearchParameters treeSearch, const float dirichletAlpha,
                            const float dirichletEpsilon, const std::size_t treeCapacity)
        : tree_search(treeSearch), dirichlet_alpha(dirichletAlpha),
          dirichlet_epsilon(dirichletEpsilon), tree_capacity(treeCapacity) {
        if (tree_capacity == 0) {
            throw std::invalid_argument("Batched search tree capacity must be positive");
        }
        if (dirichlet_alpha <= 0.0F || dirichlet_epsilon < 0.0F || dirichlet_epsilon > 1.0F) {
            throw std::invalid_argument("Batched search constants are outside their valid range");
        }
    }
};

struct BatchedInferenceParameters {
    std::size_t workers;
    std::size_t batch_size;
    std::size_t outstanding_batches_per_worker;

    BatchedInferenceParameters(const std::size_t inferenceWorkers,
                               const std::size_t inferenceBatchSize,
                               const std::size_t outstandingBatchesPerWorker)
        : workers(inferenceWorkers), batch_size(inferenceBatchSize),
          outstanding_batches_per_worker(outstandingBatchesPerWorker) {
        if (workers == 0 || batch_size == 0 || outstanding_batches_per_worker == 0 ||
            outstanding_batches_per_worker > 2) {
            throw std::invalid_argument(
                "Batched inference counts must be positive and outstanding batches at most two");
        }
    }
};
