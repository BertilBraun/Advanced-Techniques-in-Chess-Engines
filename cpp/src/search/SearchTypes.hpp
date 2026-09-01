#pragma once

#include "games/GameConcepts.hpp"
#include "search/InferenceTypes.hpp"
#include "search/SearchStopPredictor.hpp"
#include "search/SearchTree.hpp"
#include "search/tree/TreeSearchParameters.hpp"

#include <memory>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
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

enum class SearchStopReason { FixedLimit, AdditionalVisits, CapReached, LearnedEarlyStop };
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

// Learned early stopping (adaptive-stopping plan, sections 5 and 6). There is exactly one closed
// state: apply_learned == false is a flat search to the baseline — no checkpoints, no cap. An
// open policy may attenuate individual checkpoints (threshold 0 stops nothing there) but always
// caps at cap_multiple times the baseline.
struct SearchStopPolicy {
    std::vector<double> checkpoint_multiples;
    std::vector<double> thresholds;
    double movement_guard_epsilon;
    double cap_multiple;
    std::shared_ptr<const SearchStopPredictor> predictor;
    bool apply_learned;

    SearchStopPolicy()
        : checkpoint_multiples(), thresholds(), movement_guard_epsilon(1e-3), cap_multiple(2.0),
          predictor(nullptr), apply_learned(false) {}

    SearchStopPolicy(std::vector<double> checkpointMultiples, std::vector<double> stopThresholds,
                     const double movementGuardEpsilon, const double capMultiple,
                     std::shared_ptr<const SearchStopPredictor> stopPredictor,
                     const bool applyLearned)
        : checkpoint_multiples(std::move(checkpointMultiples)),
          thresholds(std::move(stopThresholds)), movement_guard_epsilon(movementGuardEpsilon),
          cap_multiple(capMultiple), predictor(std::move(stopPredictor)),
          apply_learned(applyLearned) {
        if (thresholds.size() != checkpoint_multiples.size()) {
            throw std::invalid_argument(
                "A stop policy requires one threshold per checkpoint multiple");
        }
        if (std::ranges::any_of(checkpoint_multiples, [](const double multiple) {
                return !std::isfinite(multiple) || multiple <= 0.0;
            })) {
            throw std::invalid_argument("Stop checkpoint multiples must be finite and positive");
        }
        for (std::size_t index = 1; index < checkpoint_multiples.size(); ++index) {
            if (checkpoint_multiples[index] <= checkpoint_multiples[index - 1]) {
                throw std::invalid_argument(
                    "Stop checkpoint multiples must be strictly increasing");
            }
        }
        if (!std::isfinite(cap_multiple) || cap_multiple <= 1.0) {
            throw std::invalid_argument("The stop cap multiple must be finite and above one");
        }
        if (!checkpoint_multiples.empty() && checkpoint_multiples.back() >= cap_multiple) {
            throw std::invalid_argument(
                "Stop checkpoint multiples must lie strictly below the cap multiple");
        }
        if (std::ranges::any_of(thresholds, [](const double threshold) {
                return !std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0;
            })) {
            throw std::invalid_argument("Stop thresholds must be probabilities in [0, 1]");
        }
        if (!std::isfinite(movement_guard_epsilon) || movement_guard_epsilon <= 0.0) {
            throw std::invalid_argument("The movement guard epsilon must be finite and positive");
        }
        if (apply_learned && (predictor == nullptr || checkpoint_multiples.empty())) {
            throw std::invalid_argument(
                "An applied stop policy requires a predictor and checkpoints");
        }
    }
};

struct StoppableSearchLimit {
    std::uint32_t baseline_visits;
    SearchStopPolicy policy;
    std::uint64_t model_generation;
    // Audit positions evaluate and record the stop rule but never obey it, and always search to
    // the cap even under a closed policy: they are the label source.
    bool shadow_only;

    StoppableSearchLimit()
        : baseline_visits(1), policy(), model_generation(0), shadow_only(false) {}
    StoppableSearchLimit(const std::uint32_t baselineVisits, SearchStopPolicy stopPolicy,
                         const std::uint64_t modelGeneration = 0, const bool shadowOnly = false)
        : baseline_visits(baselineVisits), policy(std::move(stopPolicy)),
          model_generation(modelGeneration), shadow_only(shadowOnly) {
        if (baseline_visits == 0) {
            throw std::invalid_argument("Stoppable search baseline must be positive");
        }
    }

    [[nodiscard]] bool searchesToCap() const noexcept {
        return policy.apply_learned || shadow_only;
    }

    [[nodiscard]] std::uint32_t capAdditionalVisits() const {
        if (!searchesToCap()) {
            return baseline_visits;
        }
        const double capped =
            std::floor(policy.cap_multiple * static_cast<double>(baseline_visits) + 0.5);
        if (capped > static_cast<double>(std::numeric_limits<std::uint32_t>::max())) {
            throw std::overflow_error("Stop cap exceeds the visit range");
        }
        return static_cast<std::uint32_t>(capped);
    }
};

using SearchLimit = std::variant<FixedSearchLimit, AdditionalSearchLimit, StoppableSearchLimit>;

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

[[nodiscard]] inline std::uint32_t maximumAdditionalVisits(const SearchLimit &limit) {
    return std::visit(
        [](const auto &selected) -> std::uint32_t {
            using Limit = std::decay_t<decltype(selected)>;
            if constexpr (std::is_same_v<Limit, FixedSearchLimit>) {
                return selected.visits;
            } else if constexpr (std::is_same_v<Limit, AdditionalSearchLimit>) {
                return selected.additional_visits;
            } else {
                return selected.capAdditionalVisits();
            }
        },
        limit);
}

struct StopCheckpointEvaluation {
    double guard_movement;
    double uncertainty;
    bool guard_passed;
    bool predictor_evaluated;
    bool would_stop;
};

// The stop rule at one checkpoint: the observational guard first (feature 5, the measured
// movement KL, must already be below the guard epsilon — a visibly moving distribution can never
// stop, whatever the predictor says), then the predictor against the checkpoint threshold.
[[nodiscard]] inline StopCheckpointEvaluation
evaluateStopRule(const SearchStopPolicy &policy, const std::size_t checkpointIndex,
                 const StopPredictorFeatures &features) {
    if (checkpointIndex >= policy.thresholds.size()) {
        throw std::invalid_argument("Stop evaluation requires a configured checkpoint index");
    }
    constexpr std::size_t movementFeatureIndex = 4;
    StopCheckpointEvaluation evaluation{
        .guard_movement = features[movementFeatureIndex],
        .uncertainty = -1.0,
        .guard_passed = features[movementFeatureIndex] < policy.movement_guard_epsilon,
        .predictor_evaluated = false,
        .would_stop = false,
    };
    if (evaluation.guard_passed && policy.predictor != nullptr) {
        evaluation.uncertainty = policy.predictor->uncertainty(features);
        evaluation.predictor_evaluated = true;
        evaluation.would_stop = evaluation.uncertainty < policy.thresholds[checkpointIndex];
    }
    return evaluation;
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
    int stop_checkpoint_index;
    std::vector<double> stop_probabilities;
    std::vector<double> guard_movements;
    std::vector<std::uint8_t> stop_verdicts;
    std::vector<StopPredictorFeatures> stop_features;
    std::uint32_t parallel_searches;
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
