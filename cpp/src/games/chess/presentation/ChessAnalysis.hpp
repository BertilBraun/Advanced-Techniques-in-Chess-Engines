#pragma once

#include "games/chess/ChessGame.hpp"
#include "search/Analysis.hpp"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using ChessAnalysis = GameAnalysis<ChessGame>;

struct ChessAnalysisCandidate {
    std::string move_uci;
    float policy_prior;
    int visits;
    float visit_share;
    std::optional<float> mean_value;
};

struct ChessAnalysisResult {
    std::string chosen_move_uci;
    float value;
    std::optional<WdlPrediction> outcome;
    std::vector<ChessAnalysisCandidate> candidates;
    int searches;
    int maximum_depth;
    std::int64_t elapsed_milliseconds;
    std::vector<std::string> principal_variation;
};

class ChessAnalysisSession {
public:
    ChessAnalysisSession(std::shared_ptr<ChessAnalysis> analysis, std::string startingFen,
                         std::vector<std::string> movesUci);

    void applyMove(const std::string &moveUci);
    [[nodiscard]] ChessAnalysisResult
    analyze(AnalysisMode mode, std::optional<int> timeLimitSeconds, std::optional<int> searchLimit);

    [[nodiscard]] std::string fen() const { return m_root.position().fen(); }
    [[nodiscard]] const std::string &startingFen() const { return m_startingFen; }
    [[nodiscard]] const std::vector<std::string> &movesUci() const { return m_movesUci; }
    [[nodiscard]] int rootVisits() const { return static_cast<int>(m_root.visits()); }

private:
    std::shared_ptr<ChessAnalysis> m_analysis;
    std::string m_startingFen;
    std::vector<std::string> m_movesUci;
    GameSearchRoot<ChessGame> m_root;

    void reconstructRoot();
};
