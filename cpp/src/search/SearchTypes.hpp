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
    float root_value;
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

struct SearchBudgetPolicy {
    static constexpr std::size_t CURVE_POINTS = SEARCH_BUDGET_CURVE_POINTS;
    std::array<double, CURVE_POINTS> multiples;
    std::array<double, CURVE_POINTS> sigma;
    double log_tau;
    double selection_threshold;
    bool apply_learned;

    SearchBudgetPolicy()
        : multiples{0.125, 0.2, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0, 1.5, 2.0, 3.0, 4.0},
          sigma{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, log_tau(0.0),
          selection_threshold(0.8), apply_learned(false) {}

    SearchBudgetPolicy(std::array<double, CURVE_POINTS> gridMultiples,
                       std::array<double, CURVE_POINTS> sigmaValues, const double logTau,
                       const double selectionThreshold, const bool applyLearned)
        : multiples(gridMultiples), sigma(sigmaValues), log_tau(logTau),
          selection_threshold(selectionThreshold), apply_learned(applyLearned) {
        if (std::ranges::any_of(multiples, [](const double multiple) {
                return !std::isfinite(multiple) || multiple <= 0.0;
            })) {
            throw std::invalid_argument("Search-budget grid multiples must be finite and positive");
        }
        for (std::size_t index = 1; index < CURVE_POINTS; ++index) {
            if (multiples[index] <= multiples[index - 1]) {
                throw std::invalid_argument(
                    "Search-budget grid multiples must be strictly increasing");
            }
        }
        if (std::ranges::none_of(multiples,
                                 [](const double multiple) { return multiple == 1.0; })) {
            throw std::invalid_argument("Search-budget grid must contain the flat multiple");
        }
        if (std::ranges::any_of(
                sigma, [](const double value) { return !std::isfinite(value) || value <= 0.0; })) {
            throw std::invalid_argument("Search-budget sigma values must be finite and positive");
        }
        if (!std::isfinite(log_tau)) {
            throw std::invalid_argument("Search-budget log tau must be finite");
        }
        if (!std::isfinite(selection_threshold) || selection_threshold <= 0.0 ||
            selection_threshold >= 1.0) {
            throw std::invalid_argument(
                "Search-budget selection threshold must lie strictly in (0, 1)");
        }
    }
};

// Running minimum from the cheapest budget upward, so more search never predicts more error.
// Sweeping the other way takes a suffix minimum, which is nondecreasing and would flatten an
// already well-formed curve to its deepest value.
[[nodiscard]] inline SearchBudgetCurvePrediction
projectNonIncreasing(SearchBudgetCurvePrediction values) {
    for (std::size_t index = 1; index < values.size(); ++index) {
        values[index] = std::min(values[index], values[index - 1]);
    }
    return values;
}

[[nodiscard]] inline double standardNormalCdf(const double value) {
    return 0.5 * (1.0 + std::erf(value / std::sqrt(2.0)));
}

// Lowest grid point whose projected predicted log KL is confidently below log tau; the deepest
// point when none qualifies.
[[nodiscard]] inline std::size_t selectBudgetIndex(const SearchBudgetPolicy &policy,
                                                   const SearchBudgetCurvePrediction &prediction) {
    if (std::ranges::any_of(prediction, [](const float value) { return !std::isfinite(value); })) {
        throw std::invalid_argument("Search-budget curve predictions must be finite");
    }
    const SearchBudgetCurvePrediction projected = projectNonIncreasing(prediction);
    for (std::size_t index = 0; index < projected.size(); ++index) {
        const double probability = standardNormalCdf(
            (policy.log_tau - static_cast<double>(projected[index])) / policy.sigma[index]);
        if (probability > policy.selection_threshold) {
            return index;
        }
    }
    return projected.size() - 1;
}

struct PredictedSearchBudgetLimit {
    std::uint32_t baseline_visits;
    SearchBudgetPolicy policy;

    PredictedSearchBudgetLimit() : baseline_visits(1), policy() {}
    PredictedSearchBudgetLimit(const std::uint32_t baselineVisits, SearchBudgetPolicy budgetPolicy)
        : baseline_visits(baselineVisits), policy(std::move(budgetPolicy)) {
        if (baseline_visits == 0) {
            throw std::invalid_argument("Predicted search-budget baseline must be positive");
        }
    }
};

using SearchLimit =
    std::variant<FixedSearchLimit, AdditionalSearchLimit, PredictedSearchBudgetLimit>;

[[nodiscard]] inline std::uint32_t searchParallelism(const std::uint32_t additionalVisits) {
    if (additionalVisits == 0) {
        throw std::invalid_argument("Assigned additional visits must be positive");
    }
    const std::uint32_t targetRounds = (additionalVisits + 199U) / 200U;
    std::uint32_t parallelSearches = 2;
    while (parallelSearches < targetRounds && parallelSearches < 16U) {
        parallelSearches *= 2U;
    }
    return std::min(parallelSearches, 16U);
}

struct AssignedSearchBudget {
    std::uint32_t additional_visits;
    int selected_index;
};

class SearchBudgetAllocator {
public:
    [[nodiscard]] AssignedSearchBudget assign(const PredictedSearchBudgetLimit &limit,
                                              const SearchBudgetCurvePrediction &prediction) {
        if (!limit.policy.apply_learned) {
            return {.additional_visits = limit.baseline_visits, .selected_index = -1};
        }
        constexpr std::uint64_t maximumBaselineMultiple = 8;
        const std::uint64_t maximumVisits =
            static_cast<std::uint64_t>(limit.baseline_visits) * maximumBaselineMultiple;
        if (maximumVisits > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Predicted search budget exceeds the visit range");
        }
        const std::size_t selectedIndex = selectBudgetIndex(limit.policy, prediction);
        const double ideal =
            static_cast<double>(limit.baseline_visits) * limit.policy.multiples[selectedIndex];
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
        return {.additional_visits = assigned, .selected_index = static_cast<int>(selectedIndex)};
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
    SearchBudgetCurvePrediction predicted_budget_curve;
    int selected_budget_index;
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
    std::size_t initial_tree_capacity;
    std::size_t maximum_tree_capacity;

    BatchedSearchParameters(TreeSearchParameters treeSearch, const float dirichletAlpha,
                            const float dirichletEpsilon, const std::size_t initialTreeCapacity,
                            const std::size_t maximumTreeCapacity)
        : tree_search(treeSearch), dirichlet_alpha(dirichletAlpha),
          dirichlet_epsilon(dirichletEpsilon), initial_tree_capacity(initialTreeCapacity),
          maximum_tree_capacity(maximumTreeCapacity) {
        if (initial_tree_capacity == 0 || maximum_tree_capacity < initial_tree_capacity) {
            throw std::invalid_argument(
                "Batched search tree capacities must be positive and ordered");
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
