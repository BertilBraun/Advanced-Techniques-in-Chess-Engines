#include "games/chess/ChessAnalysisEngine.hpp"

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

std::vector<CandidateAnalysis> gatherPolicyCandidates(const ChessInferenceResult &inferenceResult) {
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

ChessAnalysisParameters::ChessAnalysisParameters(
    const std::uint32_t parallelSearches, const float explorationConstant,
    const std::size_t inferenceWorkers, const std::size_t inferenceBatchSize,
    const std::size_t outstandingBatchesPerWorker)
    : parallel_searches(parallelSearches), exploration_constant(explorationConstant),
      inference_workers(inferenceWorkers), inference_batch_size(inferenceBatchSize),
      outstanding_batches_per_worker(outstandingBatchesPerWorker) {
    if (parallel_searches == 0 || !std::isfinite(exploration_constant) ||
        exploration_constant <= 0.0F) {
        throw std::invalid_argument("Chess analysis search parameters are invalid");
    }
}

ChessAnalysisEngine::ChessAnalysisEngine(
    const InferenceRuntimeParameters &runtimeParameters,
    const ChessAnalysisParameters &parameters)
    : m_parameters(parameters),
      m_search(runtimeParameters.model_path, runtimeParameters.device,
               runtimeParameters.device_id,
               BatchedInferenceParameters(parameters.inference_workers,
                                          parameters.inference_batch_size,
                                          parameters.outstanding_batches_per_worker),
               BatchedSearchParameters(parameters.parallel_searches,
                                       parameters.exploration_constant, 0, 1.0F, 0.0F, 1'024),
               0, false, 1.0F) {}

std::shared_ptr<ChessAnalysisSession>
ChessAnalysisEngine::newSession(const std::string &startingFen,
                                const std::vector<std::string> &movesUci) {
    return std::make_shared<ChessAnalysisSession>(shared_from_this(), startingFen, movesUci);
}

ChessAnalysisSession::ChessAnalysisSession(std::shared_ptr<ChessAnalysisEngine> engine,
                                           std::string startingFen,
                                           std::vector<std::string> movesUci)
    : m_engine(std::move(engine)), m_startingFen(std::move(startingFen)),
      m_movesUci(std::move(movesUci)),
      m_root(m_engine->m_search.newRoot(Board::replay(m_startingFen, m_movesUci),
                                        std::numeric_limits<std::uint32_t>::max())) {}

void ChessAnalysisSession::reconstructRoot() {
    m_root = m_engine->m_search.newRoot(Board::replay(m_startingFen, m_movesUci),
                                        std::numeric_limits<std::uint32_t>::max());
}

void ChessAnalysisSession::applyMove(const std::string &moveUci) {
    if (m_root.position().isGameOver()) {
        throw std::invalid_argument("Cannot apply move after game over: " + moveUci);
    }
    const Move move = m_root.position().legalMoveFromUci(moveUci);
    const ChessSearchNode &root = m_root.tree().root();
    if (root.expanded()) {
        for (std::uint32_t index = 0; index < root.children.size(); ++index) {
            if (root.children[index].action == move) {
                m_root.play(ChessGameContract::actionId(move, m_root.position()));
                m_movesUci.push_back(moveUci);
                return;
            }
        }
    }

    m_movesUci.push_back(moveUci);
    reconstructRoot();
}

AnalysisResult ChessAnalysisSession::analyze(const AnalysisMode mode,
                                             const std::optional<int> timeLimitSeconds,
                                             const std::optional<int> searchLimit) {
    const auto startedAt = std::chrono::steady_clock::now();
    if (m_root.position().isGameOver()) {
        throw std::invalid_argument("Cannot analyze a terminal position");
    }

    if (mode == AnalysisMode::Policy) {
        const ChessInferenceResult inferenceResult =
            m_engine->m_search.evaluate({m_root.position()}).front();
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

    std::uint64_t completedSearches = 0;
    const std::optional deadline =
        timeLimitSeconds.has_value()
            ? std::optional{startedAt + std::chrono::seconds(*timeLimitSeconds)}
            : std::nullopt;
    do {
        const std::uint32_t remaining = searchLimit.has_value()
                                            ? static_cast<std::uint32_t>(*searchLimit) -
                                                  static_cast<std::uint32_t>(completedSearches)
                                            : m_engine->m_parameters.parallel_searches;
        const std::uint32_t chunk =
            std::min(remaining, m_engine->m_parameters.parallel_searches);
        const GameSearchBatchResult batch = m_engine->m_search.searchDetailed(
            {{m_root, m_root.visits() + chunk, false, true}});
        completedSearches += batch.simulations_completed;
    } while ((!searchLimit.has_value() || completedSearches < *searchLimit) &&
             (!deadline.has_value() || std::chrono::steady_clock::now() < *deadline));
    const ChessSearchTree &tree = m_root.tree();
    std::vector<CandidateAnalysis> candidates = gatherMctsCandidates(tree);
    if (candidates.empty()) {
        throw std::runtime_error("MCTS returned no legal candidates");
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - startedAt);
    return {candidates.front().move_uci,
            m_root.visits() == 0 ? 0.0F
                                 : m_root.tree().root().value_sum /
                                       static_cast<float>(m_root.visits()),
            m_root.tree().root().network_outcome,
            std::move(candidates),
            static_cast<int>(completedSearches),
            tree.maximumDepth(),
            elapsed.count(),
            principalVariation(tree)};
}
