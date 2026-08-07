#pragma once

#include "search/InferenceConfiguration.hpp"
#include "search/SearchEngine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

enum class AnalysisMode { Policy, Mcts };

struct AnalysisParameters {
    std::uint32_t parallel_searches;
    float exploration_constant;
    BatchedInferenceParameters inference;

    AnalysisParameters(std::uint32_t parallelSearches, float explorationConstant,
                       BatchedInferenceParameters inferenceParameters)
        : parallel_searches(parallelSearches), exploration_constant(explorationConstant),
          inference(std::move(inferenceParameters)) {
        if (parallel_searches == 0 || !std::isfinite(exploration_constant) ||
            exploration_constant <= 0.0F) {
            throw std::invalid_argument("Analysis search parameters are invalid");
        }
    }
};

struct AnalysisCandidate {
    int action_id;
    float policy_prior;
    std::uint32_t visits;
    float visit_share;
    std::optional<float> mean_value;
};

struct GameAnalysisResult {
    int chosen_action_id;
    float value;
    std::optional<WdlPrediction> outcome;
    std::vector<AnalysisCandidate> candidates;
    std::uint64_t searches;
    int maximum_depth;
    std::int64_t elapsed_milliseconds;
    std::vector<int> principal_variation;
};

template <typename Game> class GameAnalysis {
public:
    using Position = typename Game::Position;
    using Root = GameSearchRoot<Game>;
    using Tree = GameSearchTree<Game>;
    using Node = GameSearchNode<Game>;
    using Edge = GameSearchEdge<typename Game::Action>;

    GameAnalysis(const InferenceConfiguration &runtimeParameters,
                 const AnalysisParameters &parameters)
        : m_parameters(parameters),
          m_search(runtimeParameters.model_path, runtimeParameters.device,
                   runtimeParameters.device_id, parameters.inference,
                   BatchedSearchParameters(parameters.parallel_searches,
                                           parameters.exploration_constant, 0, 1.0F, 0.0F, 1'024),
                   0, false, 1.0F) {}

    [[nodiscard]] Root newRoot(Position position) {
        return m_search.newRoot(std::move(position), std::numeric_limits<std::uint32_t>::max());
    }

    [[nodiscard]] GameAnalysisResult analyzePolicy(const Root &root) {
        rejectTerminal(root);
        const auto startedAt = std::chrono::steady_clock::now();
        const SearchInferenceResult<Game> inference = m_search.evaluate({root.position()}).front();
        std::vector<AnalysisCandidate> candidates = policyCandidates(inference, root.position());
        if (candidates.empty()) {
            throw std::runtime_error("Inference returned no legal candidates");
        }
        return {
            .chosen_action_id = candidates.front().action_id,
            .value = inference.value(),
            .outcome = inference.outcome,
            .candidates = std::move(candidates),
            .searches = 0,
            .maximum_depth = 0,
            .elapsed_milliseconds = elapsedMilliseconds(startedAt),
            .principal_variation = {},
        };
    }

    [[nodiscard]] GameAnalysisResult analyzeCounted(Root &root, const std::uint32_t searches) {
        if (searches == 0) {
            throw std::invalid_argument("Analysis search count must be positive");
        }
        return analyzeSearch(root, searches, std::nullopt);
    }

    [[nodiscard]] GameAnalysisResult analyzeTimed(Root &root, const std::chrono::seconds duration) {
        if (duration < std::chrono::seconds(1) || duration > std::chrono::seconds(30)) {
            throw std::invalid_argument("Analysis duration must be between 1 and 30 seconds");
        }
        return analyzeSearch(root, std::nullopt, std::chrono::steady_clock::now() + duration);
    }

    [[nodiscard]] InferenceStatistics inferenceStatistics() const {
        return m_search.inferenceStatistics();
    }

private:
    AnalysisParameters m_parameters;
    BatchedGameSearch<Game> m_search;

    static void rejectTerminal(const Root &root) {
        if (root.isTerminal()) {
            throw std::invalid_argument("Cannot analyze a terminal position");
        }
    }

    [[nodiscard]] static std::int64_t
    elapsedMilliseconds(const std::chrono::steady_clock::time_point startedAt) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - startedAt)
            .count();
    }

    [[nodiscard]] static std::vector<AnalysisCandidate>
    policyCandidates(const SearchInferenceResult<Game> &inference, const Position &position) {
        std::vector<AnalysisCandidate> candidates;
        candidates.reserve(inference.actions.size());
        for (const auto &[action, prior] : inference.actions) {
            candidates.push_back({
                .action_id = Game::actionId(action, position),
                .policy_prior = prior,
                .visits = 0,
                .visit_share = 0.0F,
                .mean_value = std::nullopt,
            });
        }
        std::ranges::sort(candidates,
                          [](const AnalysisCandidate &left, const AnalysisCandidate &right) {
                              if (left.policy_prior != right.policy_prior) {
                                  return left.policy_prior > right.policy_prior;
                              }
                              return left.action_id < right.action_id;
                          });
        return candidates;
    }

    [[nodiscard]] static std::vector<AnalysisCandidate> searchCandidates(const Tree &tree) {
        const Node &root = tree.root();
        std::uint64_t totalChildVisits = 0;
        for (const Edge &child : root.children) {
            totalChildVisits += child.visits;
        }
        std::vector<AnalysisCandidate> candidates;
        candidates.reserve(root.children.size());
        for (const Edge &child : root.children) {
            const std::optional<float> meanValue =
                child.visits == 0
                    ? std::nullopt
                    : std::optional<float>{-child.value_sum / static_cast<float>(child.visits)};
            candidates.push_back({
                .action_id = Game::actionId(child.action, root.position),
                .policy_prior = child.raw_prior,
                .visits = child.visits,
                .visit_share = totalChildVisits == 0 ? 0.0F
                                                     : static_cast<float>(child.visits) /
                                                           static_cast<float>(totalChildVisits),
                .mean_value = meanValue,
            });
        }
        std::ranges::sort(candidates,
                          [](const AnalysisCandidate &left, const AnalysisCandidate &right) {
                              if (left.visits != right.visits) {
                                  return left.visits > right.visits;
                              }
                              if (left.policy_prior != right.policy_prior) {
                                  return left.policy_prior > right.policy_prior;
                              }
                              return left.action_id < right.action_id;
                          });
        return candidates;
    }

    [[nodiscard]] static std::vector<int> principalVariation(const Tree &tree) {
        std::vector<int> variation;
        std::size_t nodeIndex = tree.rootIndex();
        while (true) {
            const Node &node = tree.node(nodeIndex);
            if (!node.expanded()) {
                break;
            }
            const auto best = std::ranges::max_element(
                node.children, [&node](const Edge &left, const Edge &right) {
                    if (left.visits != right.visits) {
                        return left.visits < right.visits;
                    }
                    if (left.prior != right.prior) {
                        return left.prior < right.prior;
                    }
                    return Game::actionId(left.action, node.position) >
                           Game::actionId(right.action, node.position);
                });
            if (best->visits == 0 || !best->child_index.has_value()) {
                break;
            }
            variation.push_back(Game::actionId(best->action, node.position));
            nodeIndex = *best->child_index;
        }
        return variation;
    }

    [[nodiscard]] GameAnalysisResult
    analyzeSearch(Root &root, const std::optional<std::uint32_t> searchLimit,
                  const std::optional<std::chrono::steady_clock::time_point> deadline) {
        rejectTerminal(root);
        const auto startedAt = std::chrono::steady_clock::now();
        std::uint64_t completedSearches = 0;
        do {
            const std::uint32_t remaining =
                searchLimit.has_value()
                    ? *searchLimit - static_cast<std::uint32_t>(completedSearches)
                    : m_parameters.parallel_searches;
            const std::uint32_t chunk = std::min(remaining, m_parameters.parallel_searches);
            const GameSearchBatchResult batch = m_search.searchDetailed({GameSearchRequest<Game>{
                .root = root,
                .visit_limit = root.visits() + chunk,
                .add_root_noise = false,
                .count_root_initialization = true,
            }});
            completedSearches += batch.simulations_completed;
        } while ((!searchLimit.has_value() || completedSearches < *searchLimit) &&
                 (!deadline.has_value() || std::chrono::steady_clock::now() < *deadline));

        const Tree &tree = root.tree();
        std::vector<AnalysisCandidate> candidates = searchCandidates(tree);
        if (candidates.empty()) {
            throw std::runtime_error("MCTS returned no legal candidates");
        }
        const Node &rootNode = tree.root();
        return {
            .chosen_action_id = candidates.front().action_id,
            .value = rootNode.visits == 0
                         ? 0.0F
                         : rootNode.value_sum / static_cast<float>(rootNode.visits),
            .outcome = rootNode.network_outcome,
            .candidates = std::move(candidates),
            .searches = completedSearches,
            .maximum_depth = tree.maximumDepth(),
            .elapsed_milliseconds = elapsedMilliseconds(startedAt),
            .principal_variation = principalVariation(tree),
        };
    }
};
