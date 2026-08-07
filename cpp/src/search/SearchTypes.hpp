#pragma once

#include "games/GameConcepts.hpp"
#include "search/InferenceTypes.hpp"
#include "search/SearchTree.hpp"

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
    std::vector<GameSearchVisit> visits;
};

template <SearchGame Game> struct GameSearchRequest {
    GameSearchRoot<Game> root;
    std::uint32_t visit_limit;
    bool add_root_noise;
    bool count_root_initialization = false;
};

struct GameSearchBatchResult {
    std::vector<GameSearchResult> results;
    std::uint64_t simulations_completed;
};

struct BatchedSearchParameters {
    std::uint32_t parallel_searches;
    float exploration_constant;
    std::uint32_t minimum_root_visits;
    float dirichlet_alpha;
    float dirichlet_epsilon;
    std::size_t tree_capacity;

    BatchedSearchParameters(std::uint32_t parallelSearches, float explorationConstant,
                            std::uint32_t minimumRootVisits, float dirichletAlpha,
                            float dirichletEpsilon, std::size_t treeCapacity)
        : parallel_searches(parallelSearches), exploration_constant(explorationConstant),
          minimum_root_visits(minimumRootVisits), dirichlet_alpha(dirichletAlpha),
          dirichlet_epsilon(dirichletEpsilon), tree_capacity(treeCapacity) {
        if (parallel_searches == 0 || tree_capacity == 0) {
            throw std::invalid_argument("Batched search counts and tree capacity must be positive");
        }
        if (exploration_constant <= 0.0F || dirichlet_alpha <= 0.0F || dirichlet_epsilon < 0.0F ||
            dirichlet_epsilon > 1.0F) {
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
