#pragma once

#include "InferenceRuntime.hpp"
#include "games/chess/ChessSearch.hpp"

#include <shared_mutex>

using ChessVisitCount = std::pair<int, int>;
using ChessVisitCounts = std::vector<ChessVisitCount>;

struct ChessSelfPlaySearchParameters {
    int parallel_searches;
    std::uint32_t full_searches;
    std::uint32_t fast_searches;
    float exploration_constant;
    float dirichlet_alpha;
    float dirichlet_epsilon;
    std::uint8_t minimum_root_visits;

    ChessSelfPlaySearchParameters(int parallelSearches, std::uint32_t fullSearches,
                                  std::uint32_t fastSearches, float explorationConstant,
                                  float dirichletAlpha, float dirichletEpsilon,
                                  std::uint8_t minimumRootVisits);

    [[nodiscard]] std::uint32_t arenaCapacity() const;
};

struct ChessSelfPlaySearchResult {
    float root_value;
    ChessVisitCounts visits;
    ChessSearchRoot root;
};

struct ChessSelfPlaySearchStatistics {
    float average_depth = 0.0F;
    float average_entropy = 0.0F;
    float average_kl_divergence = 0.0F;
    float average_policy_search_kl_divergence = 0.0F;
    float top_action_disagreement = 0.0F;
    float selected_action_prior_rank = 0.0F;
};

struct ChessSelfPlaySearchBatch {
    std::vector<ChessSelfPlaySearchResult> results;
    ChessSelfPlaySearchStatistics statistics;
    std::uint64_t simulations_completed;
};

struct ChessSelfPlaySearchRequest {
    ChessSearchRoot root;
    bool full_search;

    ChessSelfPlaySearchRequest(ChessSearchRoot root, bool fullSearch)
        : root(std::move(root)), full_search(fullSearch) {}
};

class ChessSelfPlaySearch {
public:
    ChessSelfPlaySearch(const InferenceRuntimeParameters &runtimeParameters,
                        const ChessSelfPlaySearchParameters &searchParameters,
                        BatchedInferenceParameters inferenceParameters,
                        std::uint64_t initialModelVersion = 0);
    ~ChessSelfPlaySearch();

    [[nodiscard]] ChessSelfPlaySearchBatch
    search(const std::vector<ChessSelfPlaySearchRequest> &requests,
           bool collectStatistics = false);
    [[nodiscard]] ChessSearchRoot newRoot(const std::string &fen) const;
    [[nodiscard]] ChessSearchRoot newRoot(Board board) const;
    [[nodiscard]] std::uint32_t arenaCapacity() const;
    [[nodiscard]] std::uint64_t modelVersion() const;

    [[nodiscard]] std::pair<InferenceStatistics, TimeInfo> inferenceStatistics();
    void refreshModel(std::uint64_t modelVersion, const std::string &modelPath);
    [[nodiscard]] bool
    updateSearchSchedule(const ChessSelfPlaySearchParameters &searchParameters);
    [[nodiscard]] std::vector<ChessInferenceResult>
    evaluate(const std::vector<const Board *> &boards);
    [[nodiscard]] std::vector<std::uintptr_t> workerIdentityTokens() const;

private:
    InferenceRuntimeParameters m_runtimeParameters;
    ChessSelfPlaySearchParameters m_searchParameters;
    std::uint32_t m_arenaCapacity;
    std::uint64_t m_modelVersion;
    BatchedInferenceParameters m_inferenceParameters;
    std::unique_ptr<BatchedGameSearch<ChessGameContract>> m_search;
    mutable std::shared_mutex m_operationMutex;
};
