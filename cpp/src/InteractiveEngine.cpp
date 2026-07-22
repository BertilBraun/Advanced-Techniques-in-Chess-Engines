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

std::vector<CandidateAnalysis> gatherMctsCandidates(const std::shared_ptr<EvalMCTSNode> &root) {
    const EvalMCTSNode::ChildSnapshot children = root->children();
    if (children == nullptr) {
        return {};
    }

    int totalChildVisits = 0;
    for (const std::shared_ptr<EvalMCTSNode> &child : *children) {
        totalChildVisits += child->number_of_visits.load(std::memory_order_relaxed);
    }

    std::vector<CandidateAnalysis> candidates;
    candidates.reserve(children->size());
    for (const std::shared_ptr<EvalMCTSNode> &child : *children) {
        const int visits = child->number_of_visits.load(std::memory_order_relaxed);
        const std::optional<float> meanValue =
            visits == 0 ? std::nullopt
                        : std::optional<float>{-child->result_sum.load(std::memory_order_relaxed) /
                                               static_cast<float>(visits)};
        candidates.push_back({toString(child->moveToGetHere), child->policy, visits,
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

std::vector<std::string> principalVariation(const std::shared_ptr<EvalMCTSNode> &root) {
    std::vector<std::string> variation;
    std::shared_ptr<EvalMCTSNode> node = root;
    while (true) {
        const EvalMCTSNode::ChildSnapshot children = node->children();
        if (children == nullptr || children->empty()) {
            break;
        }
        const auto best =
            std::ranges::max_element(*children, [](const std::shared_ptr<EvalMCTSNode> &left,
                                                   const std::shared_ptr<EvalMCTSNode> &right) {
                const int leftVisits = left->number_of_visits.load(std::memory_order_relaxed);
                const int rightVisits = right->number_of_visits.load(std::memory_order_relaxed);
                if (leftVisits != rightVisits) {
                    return leftVisits < rightVisits;
                }
                if (left->policy != right->policy) {
                    return left->policy < right->policy;
                }
                return toString(left->moveToGetHere) > toString(right->moveToGetHere);
            });
        if ((*best)->number_of_visits.load(std::memory_order_relaxed) == 0) {
            break;
        }
        variation.push_back(toString((*best)->moveToGetHere));
        node = *best;
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
    m_root = EvalMCTSNode::createRoot(replayMoves(m_startingFen, m_movesUci));
}

void InteractiveGame::applyMove(const std::string &moveUci) {
    if (m_root->isTerminal()) {
        throw std::invalid_argument("Cannot apply move after game over: " + moveUci);
    }
    const Move move = findLegalMove(m_root->board(), moveUci);
    const EvalMCTSNode::ChildSnapshot children = m_root->children();
    if (children != nullptr) {
        for (size_t index = 0; index < children->size(); ++index) {
            if ((*children)[index]->moveToGetHere == move) {
                m_root = m_root->makeNewRoot(index);
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
    if (m_root->isTerminal()) {
        throw std::invalid_argument("Cannot analyze a terminal position");
    }

    if (mode == AnalysisMode::Policy) {
        const InferenceResult inferenceResult = m_engine->m_search.evaluate(m_root->board());
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

    EvalMCTSResult searchResult =
        timeLimitSeconds.has_value()
            ? m_engine->m_search.evalSearchUntil(
                  m_root, startedAt + std::chrono::seconds(*timeLimitSeconds), searchLimit)
            : m_engine->m_search.evalSearch(m_root, *searchLimit);
    std::vector<CandidateAnalysis> candidates = gatherMctsCandidates(m_root);
    if (candidates.empty()) {
        throw std::runtime_error("MCTS returned no legal candidates");
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - startedAt);
    return {candidates.front().move_uci,
            searchResult.result,
            m_root->networkOutcome,
            std::move(candidates),
            searchResult.completed_searches,
            m_root->maxDepth(),
            elapsed.count(),
            principalVariation(m_root)};
}
