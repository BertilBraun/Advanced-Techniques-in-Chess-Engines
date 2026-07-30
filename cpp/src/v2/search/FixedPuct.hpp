#pragma once

#include "common.hpp"
#include "games/game_concepts.hpp"
#include "inference/InferenceTypes.hpp"
#include "search/SearchConfiguration.hpp"
#include "search/SearchTelemetry.hpp"
#include "search/SeededRandom.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace az::v2::search {

template <typename TerminalValue, typename State>
concept TerminalValueProvider = requires(const TerminalValue provider, const State &state) {
    { provider(state) } -> std::same_as<std::optional<double>>;
};

template <games::GameState State>
    requires std::integral<typename State::action_type>
class FixedPuctSearch {
public:
    using action_type = typename State::action_type;
    using encoding_type = typename State::encoding_type;

    template <typename Evaluator, typename TerminalValue>
        requires inference::SynchronousEvaluator<Evaluator, encoding_type> &&
                 TerminalValueProvider<TerminalValue, State>
    [[nodiscard]] static SearchResult<action_type>
    run(const State &initial_state, Evaluator &evaluator, TerminalValue terminal_value,
        const FixedPuctConfiguration &configuration) {
        configuration.validate();
        SearchContext<Evaluator, TerminalValue> context(evaluator, std::move(terminal_value),
                                                        configuration);
        return context.run(initial_state);
    }

private:
    struct Node;

    struct Child {
        action_type action;
        double prior;
        std::unique_ptr<Node> node;
    };

    struct Node {
        explicit Node(State initial_state) : state(std::move(initial_state)) {}

        State state;
        bool expanded = false;
        int32 visits = 0;
        double valueSum = 0.0;
        double expansionValue = 0.0;
        std::optional<double> censoredTerminalValue;
        std::vector<Child> children;

        [[nodiscard]] double meanValue() const {
            return visits == 0 ? 0.0 : valueSum / static_cast<double>(visits);
        }
    };

    template <typename Evaluator, typename TerminalValue> class SearchContext {
    public:
        SearchContext(Evaluator &evaluator, TerminalValue terminal_value,
                      const FixedPuctConfiguration &configuration)
            : _evaluator(evaluator), _terminalValue(std::move(terminal_value)),
              _configuration(configuration), _rootNoiseRandom(configuration.rootNoiseSeed),
              _actionSamplingRandom(configuration.actionSamplingSeed) {}

        [[nodiscard]] SearchResult<action_type> run(const State &initial_state) {
            if (initial_state.isTerminal()) {
                return terminalResult(initial_state);
            }

            Node root(initial_state);
            (void) expand(root, true);
            _initialRootFpu = firstPlayUrgency(root);
            for (int64 simulation = 0; simulation < _configuration.simulationCap; ++simulation) {
                simulate(root);
                capturePrefixTrace(root);
                if (root.visits < _configuration.simulationCap && shouldStopAdaptively(root)) {
                    return completedResult(root, SearchStopReason::AdaptiveConfidence);
                }
            }
            return completedResult(root, SearchStopReason::FullBudget);
        }

    private:
        [[nodiscard]] double expand(Node &node, bool is_root) {
            inference::InferenceResult inferenceResult = evaluate(node, is_root);
            const auto actionCount = validatedActionCount(node.state);
            const std::vector<action_type> legalActions = node.state.legalActions();
            if (legalActions.empty()) {
                throw std::logic_error("nonterminal state has no legal actions");
            }
            std::vector<double> legalPriors;
            legalPriors.reserve(legalActions.size());
            double legalMass = 0.0;
            std::vector<bool> observed(static_cast<std::size_t>(actionCount), false);
            for (const action_type action : legalActions) {
                const auto index = actionIndex(action, actionCount);
                if (observed[index]) {
                    throw std::logic_error("game returned a duplicate legal action");
                }
                observed[index] = true;
                const double prior = inferenceResult.policy[index];
                legalPriors.push_back(prior);
                legalMass += prior;
            }
            if (legalMass > 0.0) {
                if (!std::isfinite(legalMass)) {
                    throw std::invalid_argument("legal inference policy mass must be finite");
                }
                for (double &prior : legalPriors) {
                    prior /= legalMass;
                }
            } else {
                const double uniform = 1.0 / static_cast<double>(legalPriors.size());
                std::fill(legalPriors.begin(), legalPriors.end(), uniform);
            }
            if (is_root && _configuration.rootNoise.enabled) {
                const auto noise =
                    _rootNoiseRandom.dirichlet(legalPriors.size(), _configuration.rootNoise.alpha);
                for (std::size_t index = 0; index < legalPriors.size(); ++index) {
                    legalPriors[index] =
                        (1.0 - _configuration.rootNoise.fraction) * legalPriors[index] +
                        _configuration.rootNoise.fraction * noise[index];
                }
            }
            node.children.reserve(legalActions.size());
            for (std::size_t index = 0; index < legalActions.size(); ++index) {
                node.children.push_back(Child{
                    .action = legalActions[index],
                    .prior = legalPriors[index],
                    .node = nullptr,
                });
            }
            node.expanded = true;
            node.expansionValue = inferenceResult.value;
            return inferenceResult.value;
        }

        [[nodiscard]] inference::InferenceResult evaluate(Node &node, bool is_root) {
            if (is_root) {
                ++_rootInferenceRequests;
            } else {
                ++_leafInferenceRequests;
            }
            const auto requestId = _nextRequestId++;
            const inference::InferenceRequest<encoding_type> request{
                .requestId = requestId,
                .encoding = node.state.canonicalEncoding(),
                .actionCount = validatedActionCount(node.state),
            };
            inference::InferenceResult inferenceResult = _evaluator.evaluate(request);
            inference::validateResult(inferenceResult, requestId, request.actionCount);
            return inferenceResult;
        }

        void simulate(Node &root) {
            std::vector<Node *> path{&root};
            Node *node = &root;
            while (node->expanded && !node->state.isTerminal()) {
                Child &child = selectChild(*node);
                if (child.node == nullptr) {
                    State childState(node->state);
                    childState.apply(child.action);
                    child.node = std::make_unique<Node>(std::move(childState));
                }
                node = child.node.get();
                path.push_back(node);
                if (!node->expanded) {
                    break;
                }
            }

            const double value =
                node->state.isTerminal() ? terminalLeafValue(*node) : expand(*node, false);
            double backedUpValue = value;
            for (auto iterator = path.rbegin(); iterator != path.rend(); ++iterator) {
                Node &visitedNode = **iterator;
                ++visitedNode.visits;
                visitedNode.valueSum += backedUpValue;
                backedUpValue = -_configuration.backupDiscount * backedUpValue;
            }
        }

        [[nodiscard]] Child &selectChild(Node &node) const {
            const double fpu = firstPlayUrgency(node);
            const double parentScale = std::sqrt(static_cast<double>(node.visits));
            std::size_t bestIndex = 0;
            double bestScore = -std::numeric_limits<double>::infinity();
            double bestPrior = -std::numeric_limits<double>::infinity();
            for (std::size_t index = 0; index < node.children.size(); ++index) {
                const Child &child = node.children[index];
                const int32 childVisits = child.node == nullptr ? 0 : child.node->visits;
                const double actionValue =
                    childVisits == 0 ? fpu
                                     : -_configuration.backupDiscount * child.node->meanValue();
                const double exploration = _configuration.explorationConstant * child.prior *
                                           parentScale / (1.0 + static_cast<double>(childVisits));
                const double score = actionValue + exploration;
                if (score > bestScore || (score == bestScore && child.prior > bestPrior)) {
                    bestScore = score;
                    bestPrior = child.prior;
                    bestIndex = index;
                }
            }
            return node.children[bestIndex];
        }

        [[nodiscard]] double visitedChildMean(const Node &node) const {
            double total = 0.0;
            int32 count = 0;
            for (const Child &child : node.children) {
                if (child.node != nullptr && child.node->visits > 0) {
                    total += -_configuration.backupDiscount * child.node->meanValue();
                    ++count;
                }
            }
            return count == 0 ? _configuration.noVisitedChildValue
                              : total / static_cast<double>(count);
        }

        [[nodiscard]] double firstPlayUrgency(const Node &node) const {
            switch (_configuration.fpuPolicy) {
            case FpuPolicy::ParentValue:
                return parentValueEstimate(node);
            case FpuPolicy::ReducedParentValue: {
                double visitedPriorMass = 0.0;
                for (const Child &child : node.children) {
                    if (child.node != nullptr && child.node->visits > 0) {
                        visitedPriorMass += child.prior;
                    }
                }
                return parentValueEstimate(node) -
                       _configuration.fpuReduction * std::sqrt(visitedPriorMass);
            }
            case FpuPolicy::VisitedChildMean:
                return visitedChildMean(node);
            }
            throw std::logic_error("unknown FPU policy");
        }

        [[nodiscard]] static double parentValueEstimate(const Node &node) {
            return node.visits == 0 ? node.expansionValue : node.meanValue();
        }

        [[nodiscard]] bool shouldStopAdaptively(const Node &root) const {
            const AdaptiveStoppingConfiguration &stopping = _configuration.adaptiveStopping;
            if (!stopping.enabled || root.visits < stopping.minimumSimulations ||
                (root.visits - stopping.minimumSimulations) % stopping.checkIntervalSimulations !=
                    0) {
                return false;
            }
            int32 first = 0;
            int32 second = 0;
            for (const Child &child : root.children) {
                const int32 visits = child.node == nullptr ? 0 : child.node->visits;
                if (visits > first) {
                    second = first;
                    first = visits;
                } else if (visits > second) {
                    second = visits;
                }
            }
            const double simulations = static_cast<double>(root.visits);
            return static_cast<double>(first) / simulations >= stopping.requiredTopVisitFraction &&
                   static_cast<double>(first - second) / simulations >=
                       stopping.requiredTopTwoMargin;
        }

        void capturePrefixTrace(const Node &root) {
            if (!_configuration.prefixTrace.enabled ||
                _nextTraceCheckpoint >= _configuration.prefixTrace.checkpoints.size() ||
                root.visits != _configuration.prefixTrace.checkpoints[_nextTraceCheckpoint]) {
                return;
            }
            const int32 actionCount = validatedActionCount(root.state);
            std::vector<double> policy(static_cast<std::size_t>(actionCount), 0.0);
            std::vector<int32> visits(static_cast<std::size_t>(actionCount), 0);
            for (const Child &child : root.children) {
                const std::size_t index = actionIndex(child.action, actionCount);
                const int32 childVisits = child.node == nullptr ? 0 : child.node->visits;
                visits[index] = childVisits;
                policy[index] = static_cast<double>(childVisits) / static_cast<double>(root.visits);
            }
            _prefixTrace.push_back(SearchTraceSnapshot{
                .simulations = root.visits,
                .rootPolicy = std::move(policy),
                .rootVisits = std::move(visits),
                .rootValue = root.meanValue(),
            });
            ++_nextTraceCheckpoint;
        }

        [[nodiscard]] std::optional<double> validatedTerminalValue(const State &state) const {
            const std::optional<double> value = _terminalValue(state);
            if (!value.has_value()) {
                return std::nullopt;
            }
            if (!std::isfinite(*value) || *value < -1.0 || *value > 1.0) {
                throw std::invalid_argument("terminal value must be finite and in [-1, 1]");
            }
            return value;
        }

        [[nodiscard]] double terminalLeafValue(Node &node) {
            const std::optional<double> value = validatedTerminalValue(node.state);
            if (value.has_value()) {
                return *value;
            }
            if (!node.censoredTerminalValue.has_value()) {
                node.censoredTerminalValue = evaluate(node, false).value;
            }
            return *node.censoredTerminalValue;
        }

        [[nodiscard]] SearchResult<action_type> terminalResult(const State &state) const {
            const auto actionCount = validatedActionCount(state);
            return SearchResult<action_type>{
                .selectedAction = std::nullopt,
                .rootPolicy = std::vector<double>(static_cast<std::size_t>(actionCount), 0.0),
                .rootVisits = std::vector<int32>(static_cast<std::size_t>(actionCount), 0),
                .rootValue = validatedTerminalValue(state),
                .rootChildren = {},
                .telemetry =
                    SearchTelemetry{
                        .configuredCap = _configuration.simulationCap,
                        .actualSimulations = 0,
                        .budgetClass = _configuration.budgetClass,
                        .stopReason = SearchStopReason::TerminalRoot,
                        .policyTargetEligible = false,
                        .policyTargetWeight = 0.0,
                        .rootVisitCount = 0,
                        .rootInferenceRequests = 0,
                        .leafInferenceRequests = 0,
                        .totalInferenceRequests = 0,
                        .rootEntropy = 0.0,
                        .topTwoVisitMargin = 0.0,
                        .initialRootFpu = 0.0,
                    },
                .prefixTrace = {},
            };
        }

        [[nodiscard]] SearchResult<action_type> completedResult(const Node &root,
                                                                SearchStopReason stopReason) {
            const auto actionCount = validatedActionCount(root.state);
            std::vector<double> policy(static_cast<std::size_t>(actionCount), 0.0);
            std::vector<int32> visits(static_cast<std::size_t>(actionCount), 0);
            std::vector<RootChildStatistics<action_type>> children;
            children.reserve(root.children.size());
            for (const Child &child : root.children) {
                const std::size_t index = actionIndex(child.action, actionCount);
                const int32 childVisits = child.node == nullptr ? 0 : child.node->visits;
                visits[index] = childVisits;
                policy[index] = static_cast<double>(childVisits) / static_cast<double>(root.visits);
                const double actionValue =
                    childVisits == 0 ? firstPlayUrgency(root)
                                     : -_configuration.backupDiscount * child.node->meanValue();
                children.push_back(RootChildStatistics<action_type>{
                    .action = child.action,
                    .prior = child.prior,
                    .visits = childVisits,
                    .actionValue = actionValue,
                });
            }

            const std::size_t selectedIndex =
                _configuration.actionTemperature == 0.0
                    ? maximumVisitIndex(root.children)
                    : _actionSamplingRandom.sampleDiscrete(temperatureWeights(root.children));
            return SearchResult<action_type>{
                .selectedAction = root.children[selectedIndex].action,
                .rootPolicy = std::move(policy),
                .rootVisits = std::move(visits),
                .rootValue = root.meanValue(),
                .rootChildren = std::move(children),
                .telemetry =
                    SearchTelemetry{
                        .configuredCap = _configuration.simulationCap,
                        .actualSimulations = root.visits,
                        .budgetClass = _configuration.budgetClass,
                        .stopReason = stopReason,
                        .policyTargetEligible = _configuration.policyTargetWeight > 0.0,
                        .policyTargetWeight = _configuration.policyTargetWeight,
                        .rootVisitCount = root.visits,
                        .rootInferenceRequests = _rootInferenceRequests,
                        .leafInferenceRequests = _leafInferenceRequests,
                        .totalInferenceRequests = _rootInferenceRequests + _leafInferenceRequests,
                        .rootEntropy = policyEntropy(root),
                        .topTwoVisitMargin = topTwoMargin(root),
                        .initialRootFpu = _initialRootFpu,
                    },
                .prefixTrace = std::move(_prefixTrace),
            };
        }

        [[nodiscard]] static std::size_t maximumVisitIndex(const std::vector<Child> &children) {
            std::size_t bestIndex = 0;
            int32 bestVisits = -1;
            for (std::size_t index = 0; index < children.size(); ++index) {
                const int32 visits =
                    children[index].node == nullptr ? 0 : children[index].node->visits;
                if (visits > bestVisits) {
                    bestVisits = visits;
                    bestIndex = index;
                }
            }
            return bestIndex;
        }

        [[nodiscard]] std::vector<double>
        temperatureWeights(const std::vector<Child> &children) const {
            int32 maximumVisits = 0;
            for (const Child &child : children) {
                maximumVisits =
                    std::max(maximumVisits, child.node == nullptr ? 0 : child.node->visits);
            }
            if (maximumVisits <= 0) {
                throw std::logic_error("positive-temperature selection requires root visits");
            }
            const double maximumLog = std::log(static_cast<double>(maximumVisits));
            std::vector<double> weights;
            weights.reserve(children.size());
            for (const Child &child : children) {
                const int32 visits = child.node == nullptr ? 0 : child.node->visits;
                weights.push_back(
                    visits == 0 ? 0.0
                                : std::exp((std::log(static_cast<double>(visits)) - maximumLog) /
                                           _configuration.actionTemperature));
            }
            return weights;
        }

        [[nodiscard]] static double policyEntropy(const Node &root) {
            double entropy = 0.0;
            for (const Child &child : root.children) {
                const int32 visits = child.node == nullptr ? 0 : child.node->visits;
                if (visits > 0) {
                    const double probability =
                        static_cast<double>(visits) / static_cast<double>(root.visits);
                    entropy -= probability * std::log(probability);
                }
            }
            return entropy;
        }

        [[nodiscard]] static double topTwoMargin(const Node &root) {
            int32 first = 0;
            int32 second = 0;
            for (const Child &child : root.children) {
                const int32 visits = child.node == nullptr ? 0 : child.node->visits;
                if (visits > first) {
                    second = first;
                    first = visits;
                } else if (visits > second) {
                    second = visits;
                }
            }
            return static_cast<double>(first - second) / static_cast<double>(root.visits);
        }

        [[nodiscard]] static std::size_t actionIndex(action_type action, int32 actionCount) {
            if (std::cmp_less(action, 0) || std::cmp_greater_equal(action, actionCount)) {
                throw std::logic_error("game returned an out-of-range action");
            }
            return static_cast<std::size_t>(action);
        }

        [[nodiscard]] static int32 validatedActionCount(const State &state) {
            const action_type actionCount = state.actionCount();
            if (!std::in_range<int32>(actionCount) || std::cmp_less_equal(actionCount, 0)) {
                throw std::logic_error("game actionCount must be a positive int32 value");
            }
            return static_cast<int32>(actionCount);
        }

        Evaluator &_evaluator;
        TerminalValue _terminalValue;
        const FixedPuctConfiguration &_configuration;
        SeededRandom _rootNoiseRandom;
        SeededRandom _actionSamplingRandom;
        uint64 _nextRequestId = 0;
        int64 _rootInferenceRequests = 0;
        int64 _leafInferenceRequests = 0;
        double _initialRootFpu = 0.0;
        std::size_t _nextTraceCheckpoint = 0;
        std::vector<SearchTraceSnapshot> _prefixTrace;
    };
};

} // namespace az::v2::search
