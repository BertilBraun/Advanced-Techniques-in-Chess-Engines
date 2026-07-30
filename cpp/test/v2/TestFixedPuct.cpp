#include "games/game_concepts.hpp"
#include "inference/InferenceBatch.hpp"
#include "inference/InferenceTypes.hpp"
#include "search/FixedPuct.hpp"

#include <cmath>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

struct FixtureResult {
    bool terminal;
};

struct FixtureState {
    using action_type = std::int32_t;
    using player_type = std::int32_t;
    using termination_reason_type = std::int32_t;
    using terminal_result_type = FixtureResult;
    using encoding_type = std::int32_t;

    std::int32_t node = 0;
    std::int32_t depth = 0;
    std::int32_t player = 1;
    std::int32_t maximum_depth = 2;
    bool censored = false;

    [[nodiscard]] action_type action_count() const { return 3; }
    [[nodiscard]] std::vector<action_type> legal_actions() const {
        return is_terminal() ? std::vector<action_type>{} : std::vector<action_type>{0, 1};
    }
    [[nodiscard]] bool is_legal(action_type action) const {
        return !is_terminal() && (action == 0 || action == 1);
    }
    void apply(action_type action) {
        if (!is_legal(action)) {
            throw std::invalid_argument("illegal fixture action");
        }
        node = node * 10 + action + 1;
        ++depth;
        player = -player;
    }
    [[nodiscard]] player_type current_player() const { return player; }
    [[nodiscard]] termination_reason_type termination_reason() const {
        return is_terminal() ? 1 : 0;
    }
    [[nodiscard]] bool is_terminal() const { return depth >= maximum_depth; }
    [[nodiscard]] terminal_result_type terminal_result() const { return {is_terminal()}; }
    [[nodiscard]] encoding_type canonical_encoding() const { return node; }
    [[nodiscard]] std::uint64_t state_hash() const {
        return static_cast<std::uint64_t>(node * 10 + depth);
    }
    [[nodiscard]] bool operator==(const FixtureState &) const = default;
};

static_assert(az::v2::games::GameState<FixtureState>);
static_assert(az::v2::search::maximum_simulation_count == std::numeric_limits<std::int32_t>::max());

struct WideActionState {
    using action_type = std::uint64_t;
    using player_type = std::int32_t;
    using termination_reason_type = std::int32_t;
    using terminal_result_type = FixtureResult;
    using encoding_type = std::int32_t;

    [[nodiscard]] action_type action_count() const {
        return static_cast<action_type>(az::v2::search::maximum_simulation_count) + 1;
    }
    [[nodiscard]] std::vector<action_type> legal_actions() const { return {}; }
    [[nodiscard]] bool is_legal(action_type) const { return false; }
    void apply(action_type) {}
    [[nodiscard]] player_type current_player() const { return 1; }
    [[nodiscard]] termination_reason_type termination_reason() const { return 1; }
    [[nodiscard]] bool is_terminal() const { return true; }
    [[nodiscard]] terminal_result_type terminal_result() const { return {.terminal = true}; }
    [[nodiscard]] encoding_type canonical_encoding() const { return 0; }
    [[nodiscard]] std::uint64_t state_hash() const { return 0; }
    [[nodiscard]] bool operator==(const WideActionState &) const = default;
};

static_assert(az::v2::games::GameState<WideActionState>);

class FixtureEvaluator {
public:
    std::vector<double> root_policy{0.1, 0.9, 100.0};
    double root_value = 0.25;
    double leaf_value = 0.5;
    std::int32_t calls = 0;
    std::uint64_t request_id_offset = 0;

    [[nodiscard]] az::v2::inference::InferenceResult
    evaluate(const az::v2::inference::InferenceRequest<std::int32_t> &request) {
        ++calls;
        return {
            .request_id = request.request_id + request_id_offset,
            .policy = request.encoding == 0 ? root_policy : std::vector<double>{0.5, 0.5, 0.0},
            .value = request.encoding == 0 ? root_value : leaf_value,
        };
    }
};

static_assert(az::v2::inference::SynchronousEvaluator<FixtureEvaluator, std::int32_t>);

az::v2::search::FixedPuctConfiguration configuration(std::int32_t simulations = 1,
                                                     double discount = 1.0,
                                                     double temperature = 0.0,
                                                     std::uint64_t action_sampling_seed = 17) {
    return {
        .simulation_cap = simulations,
        .exploration_constant = 1.5,
        .backup_discount = discount,
        .no_visited_child_value = -0.25,
        .action_temperature = temperature,
        .root_noise_seed = 23,
        .action_sampling_seed = action_sampling_seed,
        .root_noise =
            {
                .enabled = false,
                .alpha = 0.3,
                .fraction = 0.25,
            },
        .tree_reuse = false,
    };
}

std::optional<double> terminal_value(const FixtureState &state) {
    assert(state.is_terminal());
    if (state.censored) {
        return std::nullopt;
    }
    return state.node % 2 == 0 ? -1.0 : 1.0;
}

void test_masking_backup_and_exact_accounting() {
    FixtureEvaluator evaluator;
    const auto result = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, evaluator, terminal_value, configuration(1, 0.5));
    assert(evaluator.calls == 2);
    assert(result.selected_action == 1);
    assert(result.root_visits == std::vector<std::int32_t>({0, 1, 0}));
    assert(std::abs(result.root_policy[1] - 1.0) < 1e-12);
    assert(std::abs(result.root_children[0].prior - 0.1) < 1e-12);
    assert(std::abs(result.root_children[1].prior - 0.9) < 1e-12);
    assert(std::abs(result.root_value.value() - -0.25) < 1e-12);
    assert(std::abs(result.root_children[1].action_value - -0.25) < 1e-12);
    assert(result.telemetry.configured_cap == 1);
    assert(result.telemetry.actual_simulations == 1);
    assert(result.telemetry.root_visit_count == 1);
    assert(result.telemetry.root_inference_requests == 1);
    assert(result.telemetry.leaf_inference_requests == 1);
    assert(result.telemetry.total_inference_requests == 2);
    assert(result.telemetry.budget_class == az::v2::search::SearchBudgetClass::Fixed);
    assert(result.telemetry.stop_reason == az::v2::search::SearchStopReason::FullBudget);
    assert(result.telemetry.policy_target_eligible);
    assert(result.telemetry.policy_target_weight == 1.0);
}

void test_zero_legal_mass_becomes_uniform() {
    FixtureEvaluator evaluator;
    evaluator.root_policy = {0.0, 0.0, 1.0};
    const auto result = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, evaluator, terminal_value, configuration());
    assert(std::abs(result.root_children[0].prior - 0.5) < 1e-12);
    assert(std::abs(result.root_children[1].prior - 0.5) < 1e-12);
    assert(result.selected_action == 0);
}

void test_visited_child_mean_fpu() {
    FixtureEvaluator evaluator;
    evaluator.root_policy = {0.99, 0.01, 0.0};
    evaluator.leaf_value = -1.0;
    auto search_configuration = configuration(2);
    search_configuration.exploration_constant = 0.01;
    const auto result = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, evaluator, terminal_value, search_configuration);
    assert(result.root_visits[0] == 2);
    assert(result.root_visits[1] == 0);
    assert(std::abs(result.root_children[1].action_value - 1.0) < 1e-12);
}

void test_terminal_leaf_skips_inference() {
    FixtureEvaluator evaluator;
    FixtureState state{.node = 0, .depth = 0, .player = 1, .maximum_depth = 1};
    const auto result = az::v2::search::FixedPuctSearch<FixtureState>::run(
        state, evaluator, terminal_value, configuration(1, 0.75));
    assert(evaluator.calls == 1);
    assert(result.root_visits[1] == 1);
    assert(std::abs(result.root_value.value() - 0.75) < 1e-12);
    assert(result.telemetry.root_inference_requests == 1);
    assert(result.telemetry.leaf_inference_requests == 0);
    assert(result.telemetry.total_inference_requests == 1);
}

void test_censored_terminal_leaf_uses_inference_without_expansion() {
    FixtureEvaluator evaluator;
    FixtureState state{.node = 0, .depth = 0, .player = 1, .maximum_depth = 1, .censored = true};
    const auto result = az::v2::search::FixedPuctSearch<FixtureState>::run(
        state, evaluator, terminal_value, configuration(4));
    assert(evaluator.calls == 2);
    assert(std::abs(result.root_value.value() - -0.5) < 1e-12);
    assert(result.telemetry.actual_simulations == 4);
    assert(result.telemetry.root_inference_requests == 1);
    assert(result.telemetry.leaf_inference_requests == 1);
}

void test_terminal_root_is_typed_and_does_not_infer() {
    FixtureEvaluator evaluator;
    FixtureState state{.node = 11, .depth = 2, .player = 1, .maximum_depth = 2};
    const auto result = az::v2::search::FixedPuctSearch<FixtureState>::run(
        state, evaluator, terminal_value, configuration(8));
    assert(evaluator.calls == 0);
    assert(!result.selected_action.has_value());
    assert(result.root_value == 1.0);
    assert(result.telemetry.actual_simulations == 0);
    assert(result.telemetry.root_visit_count == 0);
    assert(result.telemetry.root_inference_requests == 0);
    assert(result.telemetry.leaf_inference_requests == 0);
    assert(result.telemetry.total_inference_requests == 0);
    assert(result.telemetry.stop_reason == az::v2::search::SearchStopReason::TerminalRoot);
    assert(!result.telemetry.policy_target_eligible);
    assert(result.telemetry.policy_target_weight == 0.0);
}

void test_censored_terminal_root_preserves_unknown_value() {
    FixtureEvaluator evaluator;
    FixtureState state{.node = 11, .depth = 2, .player = 1, .maximum_depth = 2, .censored = true};
    const auto result = az::v2::search::FixedPuctSearch<FixtureState>::run(
        state, evaluator, terminal_value, configuration());
    assert(evaluator.calls == 0);
    assert(!result.root_value.has_value());
    assert(!result.selected_action.has_value());
    assert(result.telemetry.total_inference_requests == 0);
}

void test_temperature_and_repeatability() {
    FixtureEvaluator first_evaluator;
    FixtureEvaluator second_evaluator;
    const auto first = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, first_evaluator, terminal_value, configuration(32, 1.0, 1e-300, 81));
    const auto second = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, second_evaluator, terminal_value, configuration(32, 1.0, 1e-300, 81));
    assert(first.selected_action == second.selected_action);
    assert(first.root_visits == second.root_visits);
    assert(first.root_policy == second.root_policy);
    assert(first.root_value == second.root_value);
    assert(first.telemetry.root_entropy == second.telemetry.root_entropy);

    FixtureEvaluator deterministic_evaluator;
    const auto deterministic = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, deterministic_evaluator, terminal_value, configuration(32));
    const auto highest = deterministic.root_visits[0] >= deterministic.root_visits[1] ? 0 : 1;
    assert(deterministic.selected_action == highest);
}

template <typename Mutation> void expect_invalid_inference(Mutation mutation) {
    FixtureEvaluator evaluator;
    mutation(evaluator);
    bool rejected = false;
    try {
        (void) az::v2::search::FixedPuctSearch<FixtureState>::run(FixtureState{}, evaluator,
                                                                  terminal_value, configuration());
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
}

void test_invalid_inference_outputs() {
    expect_invalid_inference(
        [](FixtureEvaluator &evaluator) { evaluator.root_policy = {0.5, 0.5}; });
    expect_invalid_inference([](FixtureEvaluator &evaluator) { evaluator.request_id_offset = 1; });
    expect_invalid_inference(
        [](FixtureEvaluator &evaluator) { evaluator.root_policy = {0.5, -0.1, 0.6}; });
    expect_invalid_inference([](FixtureEvaluator &evaluator) {
        evaluator.root_policy = {0.5, std::numeric_limits<double>::quiet_NaN(), 0.5};
    });
    expect_invalid_inference([](FixtureEvaluator &evaluator) {
        evaluator.root_policy = {std::numeric_limits<double>::max(),
                                 std::numeric_limits<double>::max(), 0.0};
    });
    expect_invalid_inference([](FixtureEvaluator &evaluator) {
        evaluator.root_value = std::numeric_limits<double>::infinity();
    });
    expect_invalid_inference([](FixtureEvaluator &evaluator) { evaluator.root_value = 1.1; });
}

void test_root_noise_is_seeded() {
    FixtureEvaluator first_evaluator;
    FixtureEvaluator second_evaluator;
    auto noisy = configuration(8);
    noisy.root_noise.enabled = true;
    noisy.root_noise.fraction = 0.5;
    const auto first = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, first_evaluator, terminal_value, noisy);
    const auto second = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, second_evaluator, terminal_value, noisy);
    assert(first.root_children[0].prior == second.root_children[0].prior);
    assert(first.root_visits == second.root_visits);

    FixtureEvaluator different_evaluator;
    noisy.root_noise_seed += 1;
    const auto different = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, different_evaluator, terminal_value, noisy);
    assert(first.root_children[0].prior != different.root_children[0].prior);
}

void test_random_purposes_use_independent_streams() {
    auto baseline_configuration = configuration(16, 1.0, 1.0, 1);
    baseline_configuration.root_noise.enabled = true;
    baseline_configuration.root_noise.fraction = 0.5;
    FixtureEvaluator baseline_evaluator;
    const auto baseline = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, baseline_evaluator, terminal_value, baseline_configuration);

    bool observed_different_sample = false;
    for (std::uint64_t action_seed = 2; action_seed < 64; ++action_seed) {
        auto changed_configuration = baseline_configuration;
        changed_configuration.action_sampling_seed = action_seed;
        FixtureEvaluator evaluator;
        const auto changed = az::v2::search::FixedPuctSearch<FixtureState>::run(
            FixtureState{}, evaluator, terminal_value, changed_configuration);
        assert(changed.root_visits == baseline.root_visits);
        assert(changed.root_children[0].prior == baseline.root_children[0].prior);
        observed_different_sample =
            observed_different_sample || changed.selected_action != baseline.selected_action;
    }
    assert(observed_different_sample);
}

template <typename Operation> void expect_invalid_batch(Operation operation) {
    bool rejected = false;
    try {
        operation();
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
}

void test_invalid_configuration_and_batch_contract() {
    FixtureEvaluator evaluator;
    auto invalid = configuration();
    invalid.tree_reuse = true;
    bool rejected = false;
    try {
        (void) az::v2::search::FixedPuctSearch<FixtureState>::run(FixtureState{}, evaluator,
                                                                  terminal_value, invalid);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);

    for (const double invalid_temperature :
         {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        auto invalid_float = configuration();
        invalid_float.action_temperature = invalid_temperature;
        rejected = false;
        try {
            (void) az::v2::search::FixedPuctSearch<FixtureState>::run(
                FixtureState{}, evaluator, terminal_value, invalid_float);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        assert(rejected);
    }

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    for (const double invalid_value : {nan, infinity}) {
        auto invalid_float = configuration();
        invalid_float.exploration_constant = invalid_value;
        expect_invalid_batch([&invalid_float] { invalid_float.validate(); });

        invalid_float = configuration();
        invalid_float.backup_discount = invalid_value;
        expect_invalid_batch([&invalid_float] { invalid_float.validate(); });

        invalid_float = configuration();
        invalid_float.no_visited_child_value = invalid_value;
        expect_invalid_batch([&invalid_float] { invalid_float.validate(); });

        invalid_float = configuration();
        invalid_float.root_noise.alpha = invalid_value;
        expect_invalid_batch([&invalid_float] { invalid_float.validate(); });

        invalid_float = configuration();
        invalid_float.root_noise.fraction = invalid_value;
        expect_invalid_batch([&invalid_float] { invalid_float.validate(); });
    }
    auto zero_discount = configuration();
    zero_discount.backup_discount = 0.0;
    expect_invalid_batch([&zero_discount] { zero_discount.validate(); });

    az::v2::search::SeededRandom random(1);
    expect_invalid_batch([&random, infinity] { (void) random.dirichlet(2, infinity); });

    FixtureEvaluator wide_evaluator;
    bool rejected_wide_action_count = false;
    try {
        (void) az::v2::search::FixedPuctSearch<WideActionState>::run(
            WideActionState{}, wide_evaluator,
            [](const WideActionState &) -> std::optional<double> { return 0.0; }, configuration());
    } catch (const std::logic_error &) {
        rejected_wide_action_count = true;
    }
    assert(rejected_wide_action_count);
}

void test_batch_validation_and_ordered_association() {
    using az::v2::inference::InferenceBatch;
    using az::v2::inference::InferenceBatchResult;
    const InferenceBatch<std::int32_t> batch{
        .requests =
            {
                {.request_id = 4, .encoding = 7, .action_count = 3},
                {.request_id = 5, .encoding = 8, .action_count = 2},
            },
    };
    const InferenceBatchResult valid_result{
        .results =
            {
                {.request_id = 4, .policy = {0.2, 0.3, 0.5}, .value = 0.25},
                {.request_id = 5, .policy = {0.6, 0.4}, .value = -0.5},
            },
    };
    az::v2::inference::validate_batch_result(batch, valid_result);

    expect_invalid_batch(
        [] { az::v2::inference::validate_batch(InferenceBatch<std::int32_t>{.requests = {}}); });
    expect_invalid_batch([] {
        az::v2::inference::validate_batch(InferenceBatch<std::int32_t>{
            .requests = {{.request_id = 1, .encoding = 0, .action_count = 0}}});
    });
    expect_invalid_batch([] {
        az::v2::inference::validate_batch(
            InferenceBatch<std::int32_t>{.requests = {
                                             {.request_id = 1, .encoding = 0, .action_count = 2},
                                             {.request_id = 1, .encoding = 1, .action_count = 2},
                                         }});
    });
    expect_invalid_batch([&batch] {
        const InferenceBatchResult missing{
            .results = {{.request_id = 4, .policy = {0.2, 0.3, 0.5}, .value = 0.25}}};
        az::v2::inference::validate_batch_result(batch, missing);
    });
    expect_invalid_batch([&batch, &valid_result] {
        InferenceBatchResult mismatched = valid_result;
        mismatched.results[1].request_id = 6;
        az::v2::inference::validate_batch_result(batch, mismatched);
    });
    expect_invalid_batch([&batch, &valid_result] {
        InferenceBatchResult invalid = valid_result;
        invalid.results[0].policy[0] = -1.0;
        az::v2::inference::validate_batch_result(batch, invalid);
    });
}

} // namespace

int main() {
    test_masking_backup_and_exact_accounting();
    test_zero_legal_mass_becomes_uniform();
    test_visited_child_mean_fpu();
    test_terminal_leaf_skips_inference();
    test_censored_terminal_leaf_uses_inference_without_expansion();
    test_terminal_root_is_typed_and_does_not_infer();
    test_censored_terminal_root_preserves_unknown_value();
    test_temperature_and_repeatability();
    test_invalid_inference_outputs();
    test_root_noise_is_seeded();
    test_random_purposes_use_independent_streams();
    test_invalid_configuration_and_batch_contract();
    test_batch_validation_and_ordered_association();
}
