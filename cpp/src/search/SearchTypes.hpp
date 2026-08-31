#pragma once

#include "games/GameConcepts.hpp"
#include "search/InferenceTypes.hpp"
#include "search/SearchBudgetCorrector.hpp"
#include "search/SearchTree.hpp"
#include "search/tree/TreeSearchParameters.hpp"

#include <memory>

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
    double lagrange_multiplier;
    // Null corrector applies the predicted curve unchanged (identity correction).
    std::shared_ptr<const SearchBudgetCurveCorrector> corrector;
    bool apply_learned;

    SearchBudgetPolicy()
        : multiples{0.125, 0.2, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0, 1.5, 2.0}, lagrange_multiplier(0.0),
          corrector(nullptr), apply_learned(false) {}

    SearchBudgetPolicy(std::array<double, CURVE_POINTS> gridMultiples,
                       const double lagrangeMultiplier,
                       std::shared_ptr<const SearchBudgetCurveCorrector> curveCorrector,
                       const bool applyLearned)
        : multiples(gridMultiples), lagrange_multiplier(lagrangeMultiplier),
          corrector(std::move(curveCorrector)), apply_learned(applyLearned) {
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
        if (!std::isfinite(lagrange_multiplier) || lagrange_multiplier < 0.0) {
            throw std::invalid_argument(
                "The search-budget Lagrange multiplier must be finite and nonnegative");
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

[[nodiscard]] inline std::array<double, SearchBudgetPolicy::CURVE_POINTS>
correctBudgetCurve(const SearchBudgetPolicy &policy, const SearchBudgetCurvePrediction &prediction,
                   const SearchBudgetSelectionFeatures &features) {
    if (std::ranges::any_of(prediction, [](const float value) { return !std::isfinite(value); })) {
        throw std::invalid_argument("Search-budget curve predictions must be finite");
    }
    const std::array<double, 5> shared = {features.top_visit_share, features.policy_entropy,
                                          features.ply, features.baseline_visits,
                                          features.source_generation};
    if (std::ranges::any_of(shared, [](const double value) { return !std::isfinite(value); })) {
        throw std::invalid_argument("Search-budget selection features must be finite");
    }
    std::array<double, SearchBudgetPolicy::CURVE_POINTS> corrected{};
    if (policy.corrector == nullptr) {
        for (std::size_t index = 0; index < corrected.size(); ++index) {
            corrected[index] = static_cast<double>(prediction[index]);
        }
        return corrected;
    }
    const std::array<double, SearchBudgetPolicy::CURVE_POINTS> corrections =
        policy.corrector->correction(prediction, features);
    for (std::size_t index = 0; index < corrected.size(); ++index) {
        corrected[index] = static_cast<double>(prediction[index]) + corrections[index];
    }
    return corrected;
}

// Lagrangian selection: the grid point minimising predicted raw KL plus dual-priced spend. The
// objective works in raw KL space because the run-level quantity being minimised is a sum of KLs,
// not of logs. Ties go to the cheapest grid point.
[[nodiscard]] inline std::size_t selectBudgetIndex(const SearchBudgetPolicy &policy,
                                                   const SearchBudgetCurvePrediction &prediction,
                                                   const SearchBudgetSelectionFeatures &features) {
    std::array<double, SearchBudgetPolicy::CURVE_POINTS> projected =
        correctBudgetCurve(policy, prediction, features);
    for (std::size_t index = 1; index < projected.size(); ++index) {
        projected[index] = std::min(projected[index], projected[index - 1]);
    }
    std::size_t bestIndex = 0;
    double bestObjective = std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < projected.size(); ++index) {
        const double objective =
            std::exp(projected[index]) + policy.lagrange_multiplier * policy.multiples[index];
        if (objective < bestObjective) {
            bestObjective = objective;
            bestIndex = index;
        }
    }
    return bestIndex;
}

struct PredictedSearchBudgetLimit {
    std::uint32_t baseline_visits;
    SearchBudgetPolicy policy;
    std::uint64_t model_generation;

    PredictedSearchBudgetLimit() : baseline_visits(1), policy(), model_generation(0) {}
    PredictedSearchBudgetLimit(const std::uint32_t baselineVisits, SearchBudgetPolicy budgetPolicy,
                               const std::uint64_t modelGeneration = 0)
        : baseline_visits(baselineVisits), policy(std::move(budgetPolicy)),
          model_generation(modelGeneration) {
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
                                              const SearchBudgetCurvePrediction &prediction,
                                              const SearchBudgetSelectionFeatures &features) {
        if (!limit.policy.apply_learned) {
            return {.additional_visits = limit.baseline_visits, .selected_index = -1};
        }
        constexpr std::uint64_t maximumBaselineMultiple = 8;
        const std::uint64_t maximumVisits =
            static_cast<std::uint64_t>(limit.baseline_visits) * maximumBaselineMultiple;
        if (maximumVisits > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Predicted search budget exceeds the visit range");
        }
        const std::size_t selectedIndex = selectBudgetIndex(limit.policy, prediction, features);
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
    // Raw-prior root features: the root-time selection basis reproducible on a fresh root,
    // recorded so analysis can compare it against post-search feature values.
    float root_prior_top_share;
    float root_prior_entropy;
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
    std::uint32_t root_ply = 0;
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
