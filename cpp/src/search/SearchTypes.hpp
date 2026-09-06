#pragma once

#include "games/GameConcepts.hpp"
#include "search/InferenceTypes.hpp"
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

enum class SearchStopReason { FixedLimit, AdditionalVisits };
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

using SearchLimit = std::variant<FixedSearchLimit, AdditionalSearchLimit>;

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
            } else {
                return selected.additional_visits;
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
    float policy_surprise;
    float value_correction;
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
