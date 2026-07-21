#pragma once

#include "GameHistory.hpp"
#include "MCTS/EvalMCTS.hpp"

enum class AnalysisMode { Policy, Mcts };

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

class InteractiveGame;

class InteractiveEngine : public std::enable_shared_from_this<InteractiveEngine> {
public:
    InteractiveEngine(const InferenceClientParams &clientParameters,
                      const EvalMCTSParams &searchParameters)
        : m_search(clientParameters, searchParameters) {}

    [[nodiscard]] std::shared_ptr<InteractiveGame>
    newGame(const std::string &startingFen, const std::vector<std::string> &movesUci);

    [[nodiscard]] InferenceStatistics inferenceStatistics() {
        return m_search.inferenceStatistics();
    }

private:
    friend class InteractiveGame;
    EvalMCTS m_search;
};

class InteractiveGame {
public:
    InteractiveGame(std::shared_ptr<InteractiveEngine> engine, std::string startingFen,
                    std::vector<std::string> movesUci);

    void applyMove(const std::string &moveUci);

    [[nodiscard]] AnalysisResult analyze(AnalysisMode mode, std::optional<int> timeLimitSeconds,
                                         std::optional<int> searchLimit);

    [[nodiscard]] std::string fen() const { return m_root->board.fen(); }
    [[nodiscard]] const std::string &startingFen() const { return m_startingFen; }
    [[nodiscard]] const std::vector<std::string> &movesUci() const { return m_movesUci; }
    [[nodiscard]] int rootVisits() const {
        return m_root->number_of_visits.load(std::memory_order_relaxed);
    }

private:
    std::shared_ptr<InteractiveEngine> m_engine;
    std::string m_startingFen;
    std::vector<std::string> m_movesUci;
    std::shared_ptr<EvalMCTSNode> m_root;

    void reconstructRoot();
};
