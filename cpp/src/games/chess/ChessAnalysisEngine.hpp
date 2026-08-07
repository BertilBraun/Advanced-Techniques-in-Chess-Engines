#pragma once

#include "InferenceRuntime.hpp"
#include "games/chess/ChessSearch.hpp"

enum class AnalysisMode { Policy, Mcts };

struct ChessAnalysisParameters {
    std::uint32_t parallel_searches;
    float exploration_constant;
    std::size_t inference_workers;
    std::size_t inference_batch_size;
    std::size_t outstanding_batches_per_worker;

    ChessAnalysisParameters(std::uint32_t parallelSearches, float explorationConstant,
                            std::size_t inferenceWorkers, std::size_t inferenceBatchSize,
                            std::size_t outstandingBatchesPerWorker = 1);
};

struct CandidateAnalysis {
    std::string move_uci;
    float policy_prior;
    int visits;
    float visit_share;
    std::optional<float> mean_value;
};

struct AnalysisResult {
    std::string chosen_move_uci;
    float value;
    std::optional<WdlPrediction> outcome;
    std::vector<CandidateAnalysis> candidates;
    int searches;
    int maximum_depth;
    int64_t elapsed_milliseconds;
    std::vector<std::string> principal_variation;
};

class ChessAnalysisSession;

class ChessAnalysisEngine : public std::enable_shared_from_this<ChessAnalysisEngine> {
public:
    ChessAnalysisEngine(const InferenceRuntimeParameters &runtimeParameters,
                        const ChessAnalysisParameters &parameters);

    [[nodiscard]] std::shared_ptr<ChessAnalysisSession>
    newSession(const std::string &startingFen, const std::vector<std::string> &movesUci);

    [[nodiscard]] InferenceStatistics inferenceStatistics() {
        return m_search.inferenceStatistics();
    }

private:
    friend class ChessAnalysisSession;
    ChessAnalysisParameters m_parameters;
    BatchedGameSearch<ChessGameContract> m_search;
};

class ChessAnalysisSession {
public:
    ChessAnalysisSession(std::shared_ptr<ChessAnalysisEngine> engine, std::string startingFen,
                         std::vector<std::string> movesUci);

    void applyMove(const std::string &moveUci);

    [[nodiscard]] AnalysisResult analyze(AnalysisMode mode, std::optional<int> timeLimitSeconds,
                                         std::optional<int> searchLimit);

    [[nodiscard]] std::string fen() const { return m_root.position().fen(); }
    [[nodiscard]] const std::string &startingFen() const { return m_startingFen; }
    [[nodiscard]] const std::vector<std::string> &movesUci() const { return m_movesUci; }
    [[nodiscard]] int rootVisits() const {
        return static_cast<int>(m_root.visits());
    }

private:
    std::shared_ptr<ChessAnalysisEngine> m_engine;
    std::string m_startingFen;
    std::vector<std::string> m_movesUci;
    ChessGameSearchRoot m_root;

    void reconstructRoot();
};
