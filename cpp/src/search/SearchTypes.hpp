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
#include <vector>

// Defines requests, results, and validated runtime parameters shared by search workloads.

struct GameSearchVisit {
    int action_id;
    std::uint32_t visit_count;
};
struct GameSearchResult {
    float root_value;
    int highest_visited_child_action_id;
    std::uint32_t highest_visited_child_visit_count;
    float highest_visited_child_q;
    std::vector<GameSearchVisit> visits;
    std::vector<GameSearchVisit> policy_target_visits;
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
    std::uint32_t visit_limit;
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
