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

std::vector<CandidateAnalysis> gatherMctsCandidates(const ChessSearchTree &tree) {
    const ChessSearchNode &root = tree.node(tree.rootIndex());
    if (!root.expanded()) {
        return {};
    }

    int totalChildVisits = 0;
    for (const ChessSearchEdge &child : root.children) {
        totalChildVisits += static_cast<int>(child.visits);
    }

    std::vector<CandidateAnalysis> candidates;
    candidates.reserve(root.children.size());
    for (const ChessSearchEdge &child : root.children) {
        const int visits = static_cast<int>(child.visits);
        const std::optional<float> meanValue =
            visits == 0
                ? std::nullopt
                : std::optional<float>{-child.value_sum / static_cast<float>(visits)};
        candidates.push_back({toString(child.action), child.raw_prior, visits,
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
    candidates.reserve(inferenceResult.actions.size());
    for (const auto &[move, prior] : inferenceResult.actions) {
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

std::vector<std::string> principalVariation(const ChessSearchTree &tree) {
    std::vector<std::string> variation;
    NodeIndex nodeIndex = tree.rootIndex();
    while (true) {
        const ChessSearchNode &node = tree.node(nodeIndex);
        if (!node.expanded()) {
            break;
        }
        const auto best = std::ranges::max_element(
            node.children, [](const ChessSearchEdge &left, const ChessSearchEdge &right) {
                if (left.visits != right.visits) {
                    return left.visits < right.visits;
                }
                if (left.prior != right.prior) {
                    return left.prior < right.prior;
                }
                return toString(left.action) > toString(right.action);
            });
        if (best->visits == 0 || !best->child_index.has_value()) {
            break;
        }
        variation.push_back(toString(best->action));
        nodeIndex = *best->child_index;
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
    m_tree = std::make_unique<ChessSearchTree>(replayMoves(m_startingFen, m_movesUci), 1'024,
                                               std::numeric_limits<uint32>::max(), 1.0F);
}

void InteractiveGame::applyMove(const std::string &moveUci) {
    if (m_tree->root().position.isGameOver()) {
        throw std::invalid_argument("Cannot apply move after game over: " + moveUci);
    }
    const Move move = findLegalMove(m_tree->root().position, moveUci);
    const ChessSearchNode &root = m_tree->node(m_tree->rootIndex());
    if (root.expanded()) {
        for (std::uint32_t index = 0; index < root.children.size(); ++index) {
            if (root.children[index].action == move) {
                m_tree->rerootEdge(index);
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
    if (m_tree->root().position.isGameOver()) {
        throw std::invalid_argument("Cannot analyze a terminal position");
    }

    if (mode == AnalysisMode::Policy) {
        const InferenceResult inferenceResult =
            m_engine->m_search.evaluate(m_tree->root().position);
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
