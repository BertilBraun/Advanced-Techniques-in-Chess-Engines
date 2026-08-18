#pragma once

#include "games/GameConcepts.hpp"
#include "search/InferenceTypes.hpp"
#include "search/SearchTree.hpp"
#include "search/tree/TreeSearchParameters.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vector>

// Defines requests, results, and validated runtime parameters shared by search workloads.

struct GameSearchVisit {
    int action_id;
    std::uint32_t visit_count;

    bool operator==(const GameSearchVisit &) const = default;
};

enum class SearchStopReason { FixedLimit, Deterministic, LearnedGate, Maximum };

struct SearchCheckpoint {
    std::uint32_t visits;
    int leader_action_id;
    float top_visit_share;
    float top_two_margin;
    float root_value;
    float root_value_delta;
    bool leader_stable;
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

struct DisabledSearchCorrectionGate {};

struct SearchCorrectionGate {
    std::uint32_t visit;
    float minimum_prediction;

    SearchCorrectionGate(const std::uint32_t gateVisit, const float minimumPrediction)
        : visit(gateVisit), minimum_prediction(minimumPrediction) {
        if (visit == 0 || !std::isfinite(minimum_prediction) || minimum_prediction < 0.0F ||
            minimum_prediction > 1.0F) {
            throw std::invalid_argument("Search-correction gate is invalid");
        }
    }
};

using SearchCorrectionGateConfiguration =
    std::variant<DisabledSearchCorrectionGate, SearchCorrectionGate>;

struct AdaptiveSearchLimit {
    std::uint32_t minimum_visits;
    std::uint32_t maximum_visits;
    std::uint32_t observation_interval;
    std::uint32_t leader_stability_window;
    float root_value_tolerance;
    float initial_top_visit_share;
    float final_top_visit_share;
    float initial_top_two_margin;
    float final_top_two_margin;
    std::uint32_t threshold_relaxation_visits;
    SearchCorrectionGateConfiguration search_correction_gate;

    AdaptiveSearchLimit(std::uint32_t minimumVisits, std::uint32_t maximumVisits,
                        std::uint32_t observationInterval,
                        std::uint32_t leaderStabilityWindow, float rootValueTolerance,
                        float initialTopVisitShare, float finalTopVisitShare,
                        float initialTopTwoMargin, float finalTopTwoMargin,
                        std::uint32_t thresholdRelaxationVisits,
                        SearchCorrectionGateConfiguration searchCorrectionGate)
        : minimum_visits(minimumVisits), maximum_visits(maximumVisits),
          observation_interval(observationInterval),
          leader_stability_window(leaderStabilityWindow),
          root_value_tolerance(rootValueTolerance),
          initial_top_visit_share(initialTopVisitShare),
          final_top_visit_share(finalTopVisitShare),
          initial_top_two_margin(initialTopTwoMargin),
          final_top_two_margin(finalTopTwoMargin),
          threshold_relaxation_visits(thresholdRelaxationVisits),
          search_correction_gate(std::move(searchCorrectionGate)) {
        if (minimum_visits == 0 || maximum_visits < minimum_visits ||
            observation_interval == 0 || leader_stability_window < observation_interval ||
            leader_stability_window % observation_interval != 0 ||
            threshold_relaxation_visits == 0 || !std::isfinite(root_value_tolerance) ||
            root_value_tolerance < 0.0F || !validShare(initial_top_visit_share) ||
            !validShare(final_top_visit_share) || !validShare(initial_top_two_margin) ||
            !validShare(final_top_two_margin) ||
            final_top_visit_share > initial_top_visit_share ||
            final_top_two_margin > initial_top_two_margin) {
            throw std::invalid_argument("Adaptive search limit is invalid");
        }
        if (const auto *gate = std::get_if<SearchCorrectionGate>(&search_correction_gate);
            gate != nullptr && (gate->visit <= minimum_visits || gate->visit >= maximum_visits)) {
            throw std::invalid_argument("Search-correction gate must lie inside the adaptive range");
        }
    }

private:
    [[nodiscard]] static bool validShare(const float value) noexcept {
        return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
    }
};

using SearchLimit = std::variant<FixedSearchLimit, AdaptiveSearchLimit>;

[[nodiscard]] inline std::uint32_t maximumVisits(const SearchLimit &limit) {
    return std::visit([](const auto &selected) {
        if constexpr (std::is_same_v<std::decay_t<decltype(selected)>, FixedSearchLimit>) {
            return selected.visits;
        } else {
            return selected.maximum_visits;
        }
    }, limit);
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
    float search_correction_target;
    float predicted_search_correction;
    std::uint32_t starting_visits;
    std::uint32_t final_visits;
    SearchStopReason stop_reason;
    std::vector<SearchCheckpoint> checkpoints;
};

enum class SearchAdmission { Immediate, InitialFast, WaitingFast };

[[nodiscard]] inline std::size_t
initialFastSearchAdmissionCount(const std::size_t fastSearchCount,
                                const std::size_t fullSearchCount,
                                const std::uint32_t fullSearchBudget,
                                const std::uint32_t fastSearchBudget,
                                const std::size_t inferenceCapacity) {
    if (fastSearchCount == 0 || fullSearchCount == 0 || fastSearchBudget >= fullSearchBudget) {
        return fastSearchCount;
    }
    const std::size_t quotient = fastSearchCount / fullSearchBudget;
    const std::uint64_t remainder = fastSearchCount % fullSearchBudget;
    const std::uint64_t remainderProduct = remainder * fastSearchBudget;
    const std::size_t roundedRemainder =
        static_cast<std::size_t>((remainderProduct + fullSearchBudget - 1U) / fullSearchBudget);
    const std::size_t ratioBasedFastSearches = quotient * fastSearchBudget + roundedRemainder;
    const std::size_t capacityBasedFastSearches =
        inferenceCapacity > fullSearchCount
            ? std::min(fastSearchCount, inferenceCapacity - fullSearchCount)
            : 0;
    return std::max(ratioBasedFastSearches, capacityBasedFastSearches);
}

template <SearchGame Game> struct GameSearchRequest {
    GameSearchRoot<Game> root;
    SearchLimit limit;
    bool add_root_noise;
    bool force_root_playouts = false;
    bool count_root_initialization = false;
    SearchAdmission admission = SearchAdmission::Immediate;
};

struct GameSearchBatchResult {
    std::vector<GameSearchResult> results;
    std::uint64_t simulations_completed;
};

struct BatchedSearchParameters {
    std::uint32_t parallel_searches;
    TreeSearchParameters tree_search;
    float dirichlet_alpha;
    float dirichlet_epsilon;
    std::size_t tree_capacity;

    BatchedSearchParameters(std::uint32_t parallelSearches, TreeSearchParameters treeSearch,
                            float dirichletAlpha, float dirichletEpsilon, std::size_t treeCapacity)
        : parallel_searches(parallelSearches), tree_search(treeSearch),
          dirichlet_alpha(dirichletAlpha), dirichlet_epsilon(dirichletEpsilon),
          tree_capacity(treeCapacity) {
        if (parallel_searches == 0 || tree_capacity == 0) {
            throw std::invalid_argument("Batched search counts and tree capacity must be positive");
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

    BatchedInferenceParameters(std::size_t inferenceWorkers, std::size_t inferenceBatchSize,
                               std::size_t outstandingBatchesPerWorker)
        : workers(inferenceWorkers), batch_size(inferenceBatchSize),
          outstanding_batches_per_worker(outstandingBatchesPerWorker) {
        if (workers == 0 || batch_size == 0 || outstanding_batches_per_worker == 0 ||
            outstanding_batches_per_worker > 2) {
            throw std::invalid_argument(
                "Batched inference counts must be positive and outstanding batches at most two");
        }
    }
};
