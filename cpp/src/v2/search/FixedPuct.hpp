#pragma once

#include "games/game_concepts.hpp"
#include "inference/InferenceTypes.hpp"
#include "search/SearchConfiguration.hpp"
#include "search/SearchTelemetry.hpp"
#include "search/SeededRandom.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
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
        std::int32_t visits = 0;
        double value_sum = 0.0;
        std::optional<double> censored_terminal_value;
        std::vector<Child> children;

        [[nodiscard]] double mean_value() const {
            return visits == 0 ? 0.0 : value_sum / static_cast<double>(visits);
        }
    };

    template <typename Evaluator, typename TerminalValue> class SearchContext {
    public:
        SearchContext(Evaluator &evaluator, TerminalValue terminal_value,
                      const FixedPuctConfiguration &configuration)
            : evaluator_(evaluator), terminal_value_(std::move(terminal_value)),
              configuration_(configuration), root_noise_random_(configuration.root_noise_seed),
              action_sampling_random_(configuration.action_sampling_seed) {}

        [[nodiscard]] SearchResult<action_type> run(const State &initial_state) {
            if (initial_state.is_terminal()) {
                return terminal_result(initial_state);
            }

            Node root(initial_state);
            (void) expand(root, true);
            for (std::int64_t simulation = 0; simulation < configuration_.simulation_cap;
                 ++simulation) {
                simulate(root);
            }
            return completed_result(root);
        }

    private:
        [[nodiscard]] double expand(Node &node, bool is_root) {
            inference::InferenceResult inference_result = evaluate(node, is_root);
            const auto action_count = validated_action_count(node.state);
            const std::vector<action_type> legal_actions = node.state.legal_actions();
            if (legal_actions.empty()) {
                throw std::logic_error("nonterminal state has no legal actions");
            }
            std::vector<double> legal_priors;
            legal_priors.reserve(legal_actions.size());
            double legal_mass = 0.0;
            std::vector<bool> observed(static_cast<std::size_t>(action_count), false);
            for (const action_type action : legal_actions) {
                const auto index = action_index(action, action_count);
                if (observed[index]) {
                    throw std::logic_error("game returned a duplicate legal action");
                }
                observed[index] = true;
                const double prior = inference_result.policy[index];
                legal_priors.push_back(prior);
                legal_mass += prior;
            }
            if (legal_mass > 0.0) {
                if (!std::isfinite(legal_mass)) {
                    throw std::invalid_argument("legal inference policy mass must be finite");
                }
                for (double &prior : legal_priors) {
                    prior /= legal_mass;
                }
            } else {
                const double uniform = 1.0 / static_cast<double>(legal_priors.size());
                std::fill(legal_priors.begin(), legal_priors.end(), uniform);
            }
            if (is_root && configuration_.root_noise.enabled) {
                const auto noise = root_noise_random_.dirichlet(legal_priors.size(),
                                                                configuration_.root_noise.alpha);
                for (std::size_t index = 0; index < legal_priors.size(); ++index) {
                    legal_priors[index] =
                        (1.0 - configuration_.root_noise.fraction) * legal_priors[index] +
                        configuration_.root_noise.fraction * noise[index];
                }
            }
            node.children.reserve(legal_actions.size());
            for (std::size_t index = 0; index < legal_actions.size(); ++index) {
                node.children.push_back(Child{
                    .action = legal_actions[index],
                    .prior = legal_priors[index],
                    .node = nullptr,
                });
            }
            node.expanded = true;
            return inference_result.value;
        }

        [[nodiscard]] inference::InferenceResult evaluate(Node &node, bool is_root) {
            if (is_root) {
                ++root_inference_requests_;
            } else {
                ++leaf_inference_requests_;
            }
            const auto request_id = next_request_id_++;
            const inference::InferenceRequest<encoding_type> request{
                .request_id = request_id,
                .encoding = node.state.canonical_encoding(),
                .action_count = validated_action_count(node.state),
            };
            inference::InferenceResult inference_result = evaluator_.evaluate(request);
            inference::validate_result(inference_result, request_id, request.action_count);
            return inference_result;
        }

        void simulate(Node &root) {
            std::vector<Node *> path{&root};
            Node *node = &root;
            while (node->expanded && !node->state.is_terminal()) {
                Child &child = select_child(*node);
                if (child.node == nullptr) {
                    State child_state(node->state);
                    child_state.apply(child.action);
                    child.node = std::make_unique<Node>(std::move(child_state));
                }
                node = child.node.get();
                path.push_back(node);
                if (!node->expanded) {
                    break;
                }
            }

            const double value =
                node->state.is_terminal() ? terminal_leaf_value(*node) : expand(*node, false);
            double backed_up_value = value;
            for (auto iterator = path.rbegin(); iterator != path.rend(); ++iterator) {
                Node &visited_node = **iterator;
                ++visited_node.visits;
                visited_node.value_sum += backed_up_value;
                backed_up_value = -configuration_.backup_discount * backed_up_value;
            }
        }

        [[nodiscard]] Child &select_child(Node &node) const {
            const double fpu = visited_child_mean(node);
            const double parent_scale = std::sqrt(static_cast<double>(node.visits));
            std::size_t best_index = 0;
            double best_score = -std::numeric_limits<double>::infinity();
            double best_prior = -std::numeric_limits<double>::infinity();
            for (std::size_t index = 0; index < node.children.size(); ++index) {
                const Child &child = node.children[index];
                const std::int32_t child_visits = child.node == nullptr ? 0 : child.node->visits;
                const double action_value =
                    child_visits == 0 ? fpu
                                      : -configuration_.backup_discount * child.node->mean_value();
                const double exploration = configuration_.exploration_constant * child.prior *
                                           parent_scale / (1.0 + static_cast<double>(child_visits));
                const double score = action_value + exploration;
                if (score > best_score || (score == best_score && child.prior > best_prior)) {
                    best_score = score;
                    best_prior = child.prior;
                    best_index = index;
                }
            }
            return node.children[best_index];
        }

        [[nodiscard]] double visited_child_mean(const Node &node) const {
            double total = 0.0;
            std::int32_t count = 0;
            for (const Child &child : node.children) {
                if (child.node != nullptr && child.node->visits > 0) {
                    total += -configuration_.backup_discount * child.node->mean_value();
                    ++count;
                }
            }
            return count == 0 ? configuration_.no_visited_child_value
                              : total / static_cast<double>(count);
        }

        [[nodiscard]] std::optional<double> validated_terminal_value(const State &state) const {
            const std::optional<double> value = terminal_value_(state);
            if (!value.has_value()) {
                return std::nullopt;
            }
            if (!std::isfinite(*value) || *value < -1.0 || *value > 1.0) {
                throw std::invalid_argument("terminal value must be finite and in [-1, 1]");
            }
            return value;
        }

        [[nodiscard]] double terminal_leaf_value(Node &node) {
            const std::optional<double> value = validated_terminal_value(node.state);
            if (value.has_value()) {
                return *value;
            }
            if (!node.censored_terminal_value.has_value()) {
                node.censored_terminal_value = evaluate(node, false).value;
            }
            return *node.censored_terminal_value;
        }

        [[nodiscard]] SearchResult<action_type> terminal_result(const State &state) const {
            const auto action_count = validated_action_count(state);
            return SearchResult<action_type>{
                .selected_action = std::nullopt,
                .root_policy = std::vector<double>(static_cast<std::size_t>(action_count), 0.0),
                .root_visits = std::vector<std::int32_t>(static_cast<std::size_t>(action_count), 0),
                .root_value = validated_terminal_value(state),
                .root_children = {},
                .telemetry =
                    SearchTelemetry{
                        .configured_cap = configuration_.simulation_cap,
                        .actual_simulations = 0,
                        .budget_class = SearchBudgetClass::Fixed,
                        .stop_reason = SearchStopReason::TerminalRoot,
                        .policy_target_eligible = false,
                        .policy_target_weight = 0.0,
                        .root_visit_count = 0,
                        .root_inference_requests = 0,
                        .leaf_inference_requests = 0,
                        .total_inference_requests = 0,
                        .root_entropy = 0.0,
                        .top_two_visit_margin = 0.0,
                    },
            };
        }

        [[nodiscard]] SearchResult<action_type> completed_result(const Node &root) {
            const auto action_count = validated_action_count(root.state);
            std::vector<double> policy(static_cast<std::size_t>(action_count), 0.0);
            std::vector<std::int32_t> visits(static_cast<std::size_t>(action_count), 0);
            std::vector<RootChildStatistics<action_type>> children;
            children.reserve(root.children.size());
            for (const Child &child : root.children) {
                const std::size_t index = action_index(child.action, action_count);
                const std::int32_t child_visits = child.node == nullptr ? 0 : child.node->visits;
                visits[index] = child_visits;
                policy[index] =
                    static_cast<double>(child_visits) / static_cast<double>(root.visits);
                const double action_value =
                    child_visits == 0 ? visited_child_mean(root)
                                      : -configuration_.backup_discount * child.node->mean_value();
                children.push_back(RootChildStatistics<action_type>{
                    .action = child.action,
                    .prior = child.prior,
                    .visits = child_visits,
                    .action_value = action_value,
                });
            }

            const std::size_t selected_index =
                configuration_.action_temperature == 0.0
                    ? maximum_visit_index(root.children)
                    : action_sampling_random_.sample_discrete(temperature_weights(root.children));
            return SearchResult<action_type>{
                .selected_action = root.children[selected_index].action,
                .root_policy = std::move(policy),
                .root_visits = std::move(visits),
                .root_value = root.mean_value(),
                .root_children = std::move(children),
                .telemetry =
                    SearchTelemetry{
                        .configured_cap = configuration_.simulation_cap,
                        .actual_simulations = root.visits,
                        .budget_class = SearchBudgetClass::Fixed,
                        .stop_reason = SearchStopReason::FullBudget,
                        .policy_target_eligible = true,
                        .policy_target_weight = 1.0,
                        .root_visit_count = root.visits,
                        .root_inference_requests = root_inference_requests_,
                        .leaf_inference_requests = leaf_inference_requests_,
                        .total_inference_requests =
                            root_inference_requests_ + leaf_inference_requests_,
                        .root_entropy = policy_entropy(root),
                        .top_two_visit_margin = top_two_margin(root),
                    },
            };
        }

        [[nodiscard]] static std::size_t maximum_visit_index(const std::vector<Child> &children) {
            std::size_t best_index = 0;
            std::int32_t best_visits = -1;
            for (std::size_t index = 0; index < children.size(); ++index) {
                const std::int32_t visits =
                    children[index].node == nullptr ? 0 : children[index].node->visits;
                if (visits > best_visits) {
                    best_visits = visits;
                    best_index = index;
                }
            }
            return best_index;
        }

        [[nodiscard]] std::vector<double>
        temperature_weights(const std::vector<Child> &children) const {
            std::int32_t maximum_visits = 0;
            for (const Child &child : children) {
                maximum_visits =
                    std::max(maximum_visits, child.node == nullptr ? 0 : child.node->visits);
            }
            if (maximum_visits <= 0) {
                throw std::logic_error("positive-temperature selection requires root visits");
            }
            const double maximum_log = std::log(static_cast<double>(maximum_visits));
            std::vector<double> weights;
            weights.reserve(children.size());
            for (const Child &child : children) {
                const std::int32_t visits = child.node == nullptr ? 0 : child.node->visits;
                weights.push_back(
                    visits == 0 ? 0.0
                                : std::exp((std::log(static_cast<double>(visits)) - maximum_log) /
                                           configuration_.action_temperature));
            }
            return weights;
        }

        [[nodiscard]] static double policy_entropy(const Node &root) {
            double entropy = 0.0;
            for (const Child &child : root.children) {
                const std::int32_t visits = child.node == nullptr ? 0 : child.node->visits;
                if (visits > 0) {
                    const double probability =
                        static_cast<double>(visits) / static_cast<double>(root.visits);
                    entropy -= probability * std::log(probability);
                }
            }
            return entropy;
        }

        [[nodiscard]] static double top_two_margin(const Node &root) {
            std::int32_t first = 0;
            std::int32_t second = 0;
            for (const Child &child : root.children) {
                const std::int32_t visits = child.node == nullptr ? 0 : child.node->visits;
                if (visits > first) {
                    second = first;
                    first = visits;
                } else if (visits > second) {
                    second = visits;
                }
            }
            return static_cast<double>(first - second) / static_cast<double>(root.visits);
        }

        [[nodiscard]] static std::size_t action_index(action_type action,
                                                      std::int32_t action_count) {
            if (std::cmp_less(action, 0) || std::cmp_greater_equal(action, action_count)) {
                throw std::logic_error("game returned an out-of-range action");
            }
            return static_cast<std::size_t>(action);
        }

        [[nodiscard]] static std::int32_t validated_action_count(const State &state) {
            const action_type action_count = state.action_count();
            if (!std::in_range<std::int32_t>(action_count) ||
                std::cmp_less_equal(action_count, 0)) {
                throw std::logic_error("game action_count must be a positive int32 value");
            }
            return static_cast<std::int32_t>(action_count);
        }

        Evaluator &evaluator_;
        TerminalValue terminal_value_;
        const FixedPuctConfiguration &configuration_;
        SeededRandom root_noise_random_;
        SeededRandom action_sampling_random_;
        std::uint64_t next_request_id_ = 0;
        std::int64_t root_inference_requests_ = 0;
        std::int64_t leaf_inference_requests_ = 0;
    };
};

} // namespace az::v2::search
