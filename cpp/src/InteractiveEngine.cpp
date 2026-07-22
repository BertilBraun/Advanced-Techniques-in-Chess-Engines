#include "InteractiveEngine.hpp"

namespace {
bool candidatePreference(const CandidateAnalysis &left, const CandidateAnalysis &right) {
    if (left.visits != right.visits) {
        return left.visits > right.visits;
    }
    if (left.policy_prior != right.policy_prior) {
        return left.policy_prior > right.policy_prior;
    }
    return left.move_uci < right.move_uci;
}

std::vector<CandidateAnalysis> gatherMctsCandidates(const EvalSearchTree &tree) {
    const EvalSearchNode &root = tree.node(tree.rootIndex());
    if (!root.expanded) {
        return {};
    }

    int totalChildVisits = 0;
    for (const EvalSearchEdge &child : root.children.edges()) {
        totalChildVisits += static_cast<int>(child.statistics.visits);
    }

    std::vector<CandidateAnalysis> candidates;
    candidates.reserve(root.children.size());
    for (const EvalSearchEdge &child : root.children.edges()) {
        const int visits = static_cast<int>(child.statistics.visits);
        const std::optional<float> meanValue =
            visits == 0
                ? std::nullopt
                : std::optional<float>{-child.statistics.result_sum / static_cast<float>(visits)};
        candidates.push_back({toString(child.move), child.policy, visits,
                              totalChildVisits == 0 ? 0.0f
                                                    : static_cast<float>(visits) /
                                                          static_cast<float>(totalChildVisits),
                              meanValue});
    }
    std::ranges::sort(candidates, candidatePreference);
    return candidates;
}

std::vector<CandidateAnalysis> gatherPolicyCandidates(const InferenceResult &inferenceResult) {
    std::vector<CandidateAnalysis> candidates;
    candidates.reserve(inferenceResult.moves.size());
    for (const auto &[move, prior] : inferenceResult.moves) {
        candidates.push_back({toString(move), prior, 0, 0.0f, std::nullopt});
    }
    std::ranges::sort(candidates,
                      [](const CandidateAnalysis &left, const CandidateAnalysis &right) {
                          if (left.policy_prior != right.policy_prior) {
                              return left.policy_prior > right.policy_prior;
                          }
                          return left.move_uci < right.move_uci;
                      });
    return candidates;
}

std::vector<std::string> principalVariation(const EvalSearchTree &tree) {
    std::vector<std::string> variation;
    EvalNodeIndex nodeIndex = tree.rootIndex();
    while (true) {
        const EvalSearchNode &node = tree.node(nodeIndex);
        if (!node.expanded || node.children.empty()) {
            break;
        }
        const auto best = std::ranges::max_element(
            node.children.edges(), [](const EvalSearchEdge &left, const EvalSearchEdge &right) {
                if (left.statistics.visits != right.statistics.visits) {
                    return left.statistics.visits < right.statistics.visits;
                }
                if (left.policy != right.policy) {
                    return left.policy < right.policy;
                }
                return toString(left.move) > toString(right.move);
            });
        if (best->statistics.visits == 0 || best->child == INVALID_EVAL_NODE_INDEX) {
            break;
        }
        variation.push_back(toString(best->move));
        nodeIndex = best->child;
    }
    return variation;
}
} // namespace

std::shared_ptr<InteractiveGame>
InteractiveEngine::newGame(const std::string &startingFen,
                           const std::vector<std::string> &movesUci) {
    return std::make_shared<InteractiveGame>(shared_from_this(), startingFen, movesUci);
}

InteractiveGame::InteractiveGame(std::shared_ptr<InteractiveEngine> engine, std::string startingFen,
                                 std::vector<std::string> movesUci)
    : m_engine(std::move(engine)), m_startingFen(std::move(startingFen)),
      m_movesUci(std::move(movesUci)) {
    reconstructRoot();
}

void InteractiveGame::reconstructRoot() {
    m_tree = std::make_unique<EvalSearchTree>(replayMoves(m_startingFen, m_movesUci));
}

void InteractiveGame::applyMove(const std::string &moveUci) {
    if (m_tree->rootBoard().isGameOver()) {
        throw std::invalid_argument("Cannot apply move after game over: " + moveUci);
    }
    const Move move = findLegalMove(m_tree->rootBoard(), moveUci);
    const EvalSearchNode &root = m_tree->node(m_tree->rootIndex());
    if (root.expanded) {
        for (std::uint32_t index = 0; index < root.children.size(); ++index) {
            if (root.children[index].move == move) {
                static_cast<void>(m_tree->reroot(index));
                m_movesUci.push_back(moveUci);
                return;
            }
        }
    }

    m_movesUci.push_back(moveUci);
    reconstructRoot();
}

AnalysisResult InteractiveGame::analyze(const AnalysisMode mode,
                                        const std::optional<int> timeLimitSeconds,
                                        const std::optional<int> searchLimit) {
    const auto startedAt = std::chrono::steady_clock::now();
    if (m_tree->rootBoard().isGameOver()) {
        throw std::invalid_argument("Cannot analyze a terminal position");
    }

    if (mode == AnalysisMode::Policy) {
        const InferenceResult inferenceResult = m_engine->m_search.evaluate(m_tree->rootBoard());
        std::vector<CandidateAnalysis> candidates = gatherPolicyCandidates(inferenceResult);
        if (candidates.empty()) {
            throw std::runtime_error("Inference returned no legal candidates");
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - startedAt);
        return {candidates.front().move_uci,
                inferenceResult.value(),
                inferenceResult.outcome,
                std::move(candidates),
                0,
                0,
                elapsed.count(),
                {}};
    }

    if (!timeLimitSeconds.has_value() && !searchLimit.has_value()) {
        throw std::invalid_argument("MCTS analysis requires a time limit or search limit");
    }
    if (timeLimitSeconds.has_value() && (*timeLimitSeconds < 1 || *timeLimitSeconds > 30)) {
        throw std::invalid_argument("time_limit_seconds must be between 1 and 30");
    }
    if (searchLimit.has_value() && *searchLimit <= 0) {
        throw std::invalid_argument("search_limit must be positive");
    }

    const InteractiveSearchResult searchResult = m_engine->m_search.search(
        *m_tree,
        timeLimitSeconds.has_value()
            ? std::optional{startedAt + std::chrono::seconds(*timeLimitSeconds)}
            : std::nullopt,
        searchLimit);
    std::vector<CandidateAnalysis> candidates = gatherMctsCandidates(*m_tree);
    if (candidates.empty()) {
        throw std::runtime_error("MCTS returned no legal candidates");
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - startedAt);
    return {candidates.front().move_uci,
            searchResult.result,
            m_tree->node(m_tree->rootIndex()).network_outcome,
            std::move(candidates),
            searchResult.completed_searches,
            m_tree->maximumDepth(),
            elapsed.count(),
            principalVariation(*m_tree)};
}
