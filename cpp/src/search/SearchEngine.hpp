#pragma once

#include "search/SearchExecutor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Owns search lifecycle, retained trees, model generations, and the optimized executor.

template <typename Game> class BatchedGameSearch {
public:
    using Root = GameSearchRoot<Game>;
    using Tree = GameSearchTree<Game>;

    BatchedGameSearch(const std::string &modelPath, const InferenceDevice device,
                      const int deviceId, const BatchedInferenceParameters inferenceParameters,
                      const BatchedSearchParameters searchParameters,
                      const std::uint64_t modelGeneration, const bool resetTreesOnRefresh = true,
                      const float turnDiscount = 1.0F)
        : m_executor(modelPath, device, deviceId, inferenceParameters, searchParameters),
          m_searchParameters(searchParameters), m_modelGeneration(modelGeneration),
          m_resetTreesOnRefresh(resetTreesOnRefresh), m_turnDiscount(turnDiscount) {}

    [[nodiscard]] Root newRoot(typename Game::Position position,
                               const std::size_t maximumCapacity = 0) {
        Root root(std::move(position), m_searchParameters.tree_capacity, maximumCapacity,
                  m_turnDiscount);
        m_trees.push_back(root.sharedTree());
        return root;
    }

    [[nodiscard]] GameSearchBatchResult
    searchDetailed(const std::vector<GameSearchRequest<Game>> &requests) {
        return m_executor.searchDetailed(requests);
    }

    [[nodiscard]] std::vector<SearchInferenceResult<Game>>
    evaluate(const std::vector<typename Game::Position> &positions) {
        return m_executor.evaluate(positions);
    }

    [[nodiscard]] InferenceStatistics inferenceStatistics() const {
        return m_executor.inferenceStatistics();
    }

    void updateSearchParameters(const BatchedSearchParameters parameters) {
        m_searchParameters = parameters;
        m_executor.updateSearchParameters(parameters);
    }

    void refreshModel(const std::uint64_t modelGeneration, const std::string &modelPath) {
        if (modelGeneration <= m_modelGeneration) {
            throw std::invalid_argument("Model generation must increase during refresh");
        }
        if (!m_executor.idle()) {
            throw std::logic_error("Batched search must be idle during model refresh");
        }
        m_executor.refreshModel(modelPath);
        m_modelGeneration = modelGeneration;
        if (m_resetTreesOnRefresh) {
            resetActiveTrees();
        }
    }

    [[nodiscard]] std::uint64_t modelGeneration() const noexcept { return m_modelGeneration; }

private:
    BatchedSearchExecutor<Game> m_executor;
    BatchedSearchParameters m_searchParameters;
    std::uint64_t m_modelGeneration;
    std::vector<std::weak_ptr<Tree>> m_trees;
    bool m_resetTreesOnRefresh;
    float m_turnDiscount;

    void resetActiveTrees() {
        for (const std::weak_ptr<Tree> &tree : m_trees) {
            if (const std::shared_ptr<Tree> active = tree.lock()) {
                active->reset();
            }
        }
    }
};
