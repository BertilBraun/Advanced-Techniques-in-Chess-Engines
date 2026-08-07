#include "games/chess/ChessAnalysis.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace {
Stockfish::Move decodeAction(const Board &position, const int actionId) {
    return ChessAction::decode(actionId, position).move;
}

ChessAnalysisResult present(const GameAnalysisResult &analysis, const Board &rootPosition) {
    std::vector<ChessAnalysisCandidate> candidates;
    candidates.reserve(analysis.candidates.size());
    for (const AnalysisCandidate &candidate : analysis.candidates) {
        candidates.push_back({
            .move_uci = ChessAction(decodeAction(rootPosition, candidate.action_id)).toUci(),
            .policy_prior = candidate.policy_prior,
            .visits = static_cast<int>(candidate.visits),
            .visit_share = candidate.visit_share,
            .mean_value = candidate.mean_value,
        });
    }
    std::ranges::sort(candidates,
                      [](const ChessAnalysisCandidate &left, const ChessAnalysisCandidate &right) {
                          if (left.visits != right.visits) {
                              return left.visits > right.visits;
                          }
                          if (left.policy_prior != right.policy_prior) {
                              return left.policy_prior > right.policy_prior;
                          }
                          return left.move_uci < right.move_uci;
                      });

    Board variationPosition(rootPosition);
    std::vector<std::string> variation;
    variation.reserve(analysis.principal_variation.size());
    for (const int actionId : analysis.principal_variation) {
        const Stockfish::Move move = decodeAction(variationPosition, actionId);
        variation.push_back(ChessAction(move).toUci());
        variationPosition.makeMove(move);
    }

    return {
        .chosen_move_uci = candidates.front().move_uci,
        .value = analysis.value,
        .outcome = analysis.outcome,
        .candidates = std::move(candidates),
        .searches = static_cast<int>(analysis.searches),
        .maximum_depth = analysis.maximum_depth,
        .elapsed_milliseconds = analysis.elapsed_milliseconds,
        .principal_variation = std::move(variation),
    };
}
} // namespace

ChessAnalysisSession::ChessAnalysisSession(std::shared_ptr<ChessAnalysis> analysis,
                                           std::string startingFen,
                                           std::vector<std::string> movesUci)
    : m_analysis(std::move(analysis)), m_startingFen(std::move(startingFen)),
      m_movesUci(std::move(movesUci)),
      m_root(m_analysis->newRoot(Board::replay(m_startingFen, m_movesUci))) {}

void ChessAnalysisSession::reconstructRoot() {
    m_root = m_analysis->newRoot(Board::replay(m_startingFen, m_movesUci));
}

void ChessAnalysisSession::applyMove(const std::string &moveUci) {
    if (m_root.position().isGameOver()) {
        throw std::invalid_argument("Cannot apply move after game over: " + moveUci);
    }
    const Stockfish::Move move = m_root.position().legalMoveFromUci(moveUci);
    const GameSearchNode<ChessGameContract> &root = m_root.tree().root();
    if (root.expanded()) {
        for (const GameSearchEdge<ChessAction> &edge : root.children) {
            if (edge.action.move == move) {
                m_root.play(ChessGameContract::actionId(ChessAction(move), m_root.position()));
                m_movesUci.push_back(moveUci);
                return;
            }
        }
    }
    m_movesUci.push_back(moveUci);
    reconstructRoot();
}

ChessAnalysisResult ChessAnalysisSession::analyze(const AnalysisMode mode,
                                                  const std::optional<int> timeLimitSeconds,
                                                  const std::optional<int> searchLimit) {
    if (mode == AnalysisMode::Policy) {
        return present(m_analysis->analyzePolicy(m_root), m_root.position());
    }
    if (timeLimitSeconds.has_value() == searchLimit.has_value()) {
        throw std::invalid_argument(
            "MCTS analysis requires exactly one time limit or search limit");
    }
    if (timeLimitSeconds.has_value()) {
        return present(m_analysis->analyzeTimed(m_root, std::chrono::seconds(*timeLimitSeconds)),
                       m_root.position());
    }
    if (*searchLimit <= 0) {
        throw std::invalid_argument("search_limit must be positive");
    }
    return present(m_analysis->analyzeCounted(m_root, static_cast<std::uint32_t>(*searchLimit)),
                   m_root.position());
}
