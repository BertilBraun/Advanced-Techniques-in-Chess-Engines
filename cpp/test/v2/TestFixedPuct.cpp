#include "games/game_concepts.hpp"
#include "inference/InferenceBatch.hpp"
#include "inference/InferenceTypes.hpp"
#include "search/FixedPuct.hpp"

#include <cmath>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
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
    using action_type = int32;
    using player_type = int32;
    using termination_reason_type = int32;
    using terminal_result_type = FixtureResult;
    using encoding_type = int32;

    int32 node = 0;
    int32 depth = 0;
    int32 player = 1;
    int32 maximum_depth = 2;
    bool censored = false;

    [[nodiscard]] action_type actionCount() const { return 3; }
    [[nodiscard]] std::vector<action_type> legalActions() const {
        return isTerminal() ? std::vector<action_type>{} : std::vector<action_type>{0, 1};
    }
    [[nodiscard]] bool isLegal(action_type action) const {
        return !isTerminal() && (action == 0 || action == 1);
    }
    void apply(action_type action) {
        if (!isLegal(action)) {
            throw std::invalid_argument("illegal fixture action");
        }
        node = node * 10 + action + 1;
        ++depth;
        player = -player;
    }
    [[nodiscard]] player_type currentPlayer() const { return player; }
    [[nodiscard]] termination_reason_type terminationReason() const { return isTerminal() ? 1 : 0; }
    [[nodiscard]] bool isTerminal() const { return depth >= maximum_depth; }
    [[nodiscard]] terminal_result_type terminalResult() const { return {isTerminal()}; }
    [[nodiscard]] encoding_type canonicalEncoding() const { return node; }
    [[nodiscard]] uint64 stateHash() const { return static_cast<uint64>(node * 10 + depth); }
    [[nodiscard]] bool operator==(const FixtureState &) const = default;
};

static_assert(az::v2::games::GameState<FixtureState>);
static_assert(az::v2::search::MAXIMUM_SIMULATION_COUNT == std::numeric_limits<int32>::max());

struct WideActionState {
    using action_type = uint64;
    using player_type = int32;
    using termination_reason_type = int32;
    using terminal_result_type = FixtureResult;
    using encoding_type = int32;

    [[nodiscard]] action_type actionCount() const {
        return static_cast<action_type>(az::v2::search::MAXIMUM_SIMULATION_COUNT) + 1;
    }
    [[nodiscard]] std::vector<action_type> legalActions() const { return {}; }
    [[nodiscard]] bool isLegal(action_type) const { return false; }
    void apply(action_type) {}
    [[nodiscard]] player_type currentPlayer() const { return 1; }
    [[nodiscard]] termination_reason_type terminationReason() const { return 1; }
    [[nodiscard]] bool isTerminal() const { return true; }
    [[nodiscard]] terminal_result_type terminalResult() const { return {.terminal = true}; }
    [[nodiscard]] encoding_type canonicalEncoding() const { return 0; }
    [[nodiscard]] uint64 stateHash() const { return 0; }
    [[nodiscard]] bool operator==(const WideActionState &) const = default;
};

static_assert(az::v2::games::GameState<WideActionState>);

class FixtureEvaluator {
public:
    std::vector<double> rootPolicy{0.1, 0.9, 100.0};
    double rootValue = 0.25;
    double leaf_value = 0.5;
    int32 calls = 0;
    uint64 request_id_offset = 0;

    [[nodiscard]] az::v2::inference::InferenceResult
    evaluate(const az::v2::inference::InferenceRequest<int32> &request) {
        ++calls;
        return {
            .requestId = request.requestId + request_id_offset,
            .policy = request.encoding == 0 ? rootPolicy : std::vector<double>{0.5, 0.5, 0.0},
            .value = request.encoding == 0 ? rootValue : leaf_value,
        };
    }
};

static_assert(az::v2::inference::SynchronousEvaluator<FixtureEvaluator, int32>);

az::v2::search::FixedPuctConfiguration configuration(int32 simulations = 1, double discount = 1.0,
                                                     double temperature = 0.0,
                                                     uint64 actionSamplingSeed = 17) {
    return {
        .simulationCap = simulations,
        .explorationConstant = 1.5,
        .backupDiscount = discount,
        .noVisitedChildValue = -0.25,
        .actionTemperature = temperature,
        .rootNoiseSeed = 23,
        .actionSamplingSeed = actionSamplingSeed,
        .rootNoise =
            {
                .enabled = false,
                .alpha = 0.3,
                .fraction = 0.25,
            },
        .treeReuse = false,
    };
}

std::optional<double> terminalValue(const FixtureState &state) {
    assert(state.isTerminal());
    if (state.censored) {
        return std::nullopt;
    }
    return state.node % 2 == 0 ? -1.0 : 1.0;
}

void testMaskingBackupAndExactAccounting() {
    FixtureEvaluator evaluator;
    const auto result = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, evaluator, terminalValue, configuration(1, 0.5));
    assert(evaluator.calls == 2);
    assert(result.selectedAction == 1);
    assert(result.rootVisits == std::vector<int32>({0, 1, 0}));
    assert(std::abs(result.rootPolicy[1] - 1.0) < 1e-12);
    assert(std::abs(result.rootChildren[0].prior - 0.1) < 1e-12);
    assert(std::abs(result.rootChildren[1].prior - 0.9) < 1e-12);
    assert(std::abs(result.rootValue.value() - -0.25) < 1e-12);
    assert(std::abs(result.rootChildren[1].actionValue - -0.25) < 1e-12);
    assert(result.telemetry.configuredCap == 1);
    assert(result.telemetry.actualSimulations == 1);
    assert(result.telemetry.rootVisitCount == 1);
    assert(result.telemetry.rootInferenceRequests == 1);
    assert(result.telemetry.leafInferenceRequests == 1);
    assert(result.telemetry.totalInferenceRequests == 2);
    assert(result.telemetry.budgetClass == az::v2::search::SearchBudgetClass::Fixed);
    assert(result.telemetry.stopReason == az::v2::search::SearchStopReason::FullBudget);
    assert(result.telemetry.policyTargetEligible);
    assert(result.telemetry.policyTargetWeight == 1.0);
}

void testZeroLegalMassBecomesUniform() {
    FixtureEvaluator evaluator;
    evaluator.rootPolicy = {0.0, 0.0, 1.0};
    const auto result = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, evaluator, terminalValue, configuration());
    assert(std::abs(result.rootChildren[0].prior - 0.5) < 1e-12);
    assert(std::abs(result.rootChildren[1].prior - 0.5) < 1e-12);
    assert(result.selectedAction == 0);
}

void testVisitedChildMeanFpu() {
    FixtureEvaluator evaluator;
    evaluator.rootPolicy = {0.99, 0.01, 0.0};
    evaluator.leaf_value = -1.0;
    auto searchConfiguration = configuration(2);
    searchConfiguration.explorationConstant = 0.01;
    const auto result = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, evaluator, terminalValue, searchConfiguration);
    assert(result.rootVisits[0] == 2);
    assert(result.rootVisits[1] == 0);
    assert(std::abs(result.rootChildren[1].actionValue - 1.0) < 1e-12);
}

void testParentValueFpuUsesExpansionValueBeforeFirstBackup() {
    FixtureEvaluator evaluator;
    evaluator.rootValue = 0.625;
    auto searchConfiguration = configuration(1);
    searchConfiguration.fpuPolicy = az::v2::search::FpuPolicy::ParentValue;
    const auto result = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, evaluator, terminalValue, searchConfiguration);
    assert(result.rootVisits[0] == 0);
    assert(std::abs(result.telemetry.initialRootFpu - 0.625) < 1e-12);
}

void testReducedParentFpuUsesVisitedPolicyMass() {
    FixtureEvaluator evaluator;
    evaluator.rootPolicy = {0.1, 0.9, 0.0};
    evaluator.leaf_value = -1.0;
    auto searchConfiguration = configuration(1);
    searchConfiguration.fpuPolicy = az::v2::search::FpuPolicy::ReducedParentValue;
    searchConfiguration.fpuReduction = 0.5;
    const auto result = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, evaluator, terminalValue, searchConfiguration);
    const double expected = 1.0 - 0.5 * std::sqrt(0.9);
    assert(result.rootVisits[0] == 0);
    assert(std::abs(result.rootChildren[0].actionValue - expected) < 1e-12);
}

void testAdaptiveStoppingChecksOnlyConfiguredIntervals() {
    FixtureEvaluator evaluator;
    evaluator.rootPolicy = {0.001, 0.999, 0.0};
    auto searchConfiguration = configuration(20);
    searchConfiguration.explorationConstant = 0.01;
    searchConfiguration.adaptiveStopping = {
        .enabled = true,
        .minimumSimulations = 10,
        .checkIntervalSimulations = 4,
        .requiredTopVisitFraction = 0.75,
        .requiredTopTwoMargin = 0.5,
    };
    const auto result = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, evaluator, terminalValue, searchConfiguration);
    assert(result.telemetry.actualSimulations == 10);
    assert(result.telemetry.stopReason == az::v2::search::SearchStopReason::AdaptiveConfidence);
    assert(result.telemetry.budgetClass == az::v2::search::SearchBudgetClass::Fixed);
}

void testBudgetMetadataDoesNotChangeSearch() {
    FixtureEvaluator evaluator;
    auto searchConfiguration = configuration(3);
    searchConfiguration.budgetClass = az::v2::search::SearchBudgetClass::MixedFast;
    searchConfiguration.policyTargetWeight = 0.0;
    const auto result = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, evaluator, terminalValue, searchConfiguration);
    assert(result.telemetry.actualSimulations == 3);
    assert(result.telemetry.budgetClass == az::v2::search::SearchBudgetClass::MixedFast);
    assert(!result.telemetry.policyTargetEligible);
    assert(result.telemetry.policyTargetWeight == 0.0);
}

void testAdaptiveCadenceAtCapReportsFullBudget() {
    FixtureEvaluator evaluator;
    evaluator.rootPolicy = {0.5, 0.5, 0.0};
    auto prefixConfiguration = configuration(2);
    prefixConfiguration.explorationConstant = 0.01;
    const auto prefix = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, evaluator, terminalValue, prefixConfiguration);
    assert(prefix.rootVisits == std::vector<int32>({1, 1, 0}));

    auto searchConfiguration = configuration(3);
    searchConfiguration.explorationConstant = 0.01;
    searchConfiguration.adaptiveStopping = {
        .enabled = true,
        .minimumSimulations = 2,
        .checkIntervalSimulations = 1,
        .requiredTopVisitFraction = 0.6,
        .requiredTopTwoMargin = 0.3,
    };
    const auto result = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, evaluator, terminalValue, searchConfiguration);
    assert(result.rootVisits == std::vector<int32>({2, 1, 0}));
    assert(result.telemetry.actualSimulations == 3);
    assert(result.telemetry.topTwoVisitMargin > 0.3);
    assert(result.telemetry.stopReason == az::v2::search::SearchStopReason::FullBudget);
}

void testTerminalLeafSkipsInference() {
    FixtureEvaluator evaluator;
    FixtureState state{.node = 0, .depth = 0, .player = 1, .maximum_depth = 1};
    const auto result = az::v2::search::FixedPuctSearch<FixtureState>::run(
        state, evaluator, terminalValue, configuration(1, 0.75));
    assert(evaluator.calls == 1);
    assert(result.rootVisits[1] == 1);
    assert(std::abs(result.rootValue.value() - 0.75) < 1e-12);
    assert(result.telemetry.rootInferenceRequests == 1);
    assert(result.telemetry.leafInferenceRequests == 0);
    assert(result.telemetry.totalInferenceRequests == 1);
}

void testCensoredTerminalLeafUsesInferenceWithoutExpansion() {
    FixtureEvaluator evaluator;
    FixtureState state{.node = 0, .depth = 0, .player = 1, .maximum_depth = 1, .censored = true};
    const auto result = az::v2::search::FixedPuctSearch<FixtureState>::run(
        state, evaluator, terminalValue, configuration(4));
    assert(evaluator.calls == 2);
    assert(std::abs(result.rootValue.value() - -0.5) < 1e-12);
    assert(result.telemetry.actualSimulations == 4);
    assert(result.telemetry.rootInferenceRequests == 1);
    assert(result.telemetry.leafInferenceRequests == 1);
}

void testTerminalRootIsTypedAndDoesNotInfer() {
    FixtureEvaluator evaluator;
    FixtureState state{.node = 11, .depth = 2, .player = 1, .maximum_depth = 2};
    const auto result = az::v2::search::FixedPuctSearch<FixtureState>::run(
        state, evaluator, terminalValue, configuration(8));
    assert(evaluator.calls == 0);
    assert(!result.selectedAction.has_value());
    assert(result.rootValue == 1.0);
    assert(result.telemetry.actualSimulations == 0);
    assert(result.telemetry.rootVisitCount == 0);
    assert(result.telemetry.rootInferenceRequests == 0);
    assert(result.telemetry.leafInferenceRequests == 0);
    assert(result.telemetry.totalInferenceRequests == 0);
    assert(result.telemetry.stopReason == az::v2::search::SearchStopReason::TerminalRoot);
    assert(!result.telemetry.policyTargetEligible);
    assert(result.telemetry.policyTargetWeight == 0.0);
}

void testCensoredTerminalRootPreservesUnknownValue() {
    FixtureEvaluator evaluator;
    FixtureState state{.node = 11, .depth = 2, .player = 1, .maximum_depth = 2, .censored = true};
    const auto result = az::v2::search::FixedPuctSearch<FixtureState>::run(
        state, evaluator, terminalValue, configuration());
    assert(evaluator.calls == 0);
    assert(!result.rootValue.has_value());
    assert(!result.selectedAction.has_value());
    assert(result.telemetry.totalInferenceRequests == 0);
}

void testTemperatureAndRepeatability() {
    FixtureEvaluator firstEvaluator;
    FixtureEvaluator secondEvaluator;
    const auto first = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, firstEvaluator, terminalValue, configuration(32, 1.0, 1e-300, 81));
    const auto second = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, secondEvaluator, terminalValue, configuration(32, 1.0, 1e-300, 81));
    assert(first.selectedAction == second.selectedAction);
    assert(first.rootVisits == second.rootVisits);
    assert(first.rootPolicy == second.rootPolicy);
    assert(first.rootValue == second.rootValue);
    assert(first.telemetry.rootEntropy == second.telemetry.rootEntropy);

    FixtureEvaluator deterministicEvaluator;
    const auto deterministic = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, deterministicEvaluator, terminalValue, configuration(32));
    const auto highest = deterministic.rootVisits[0] >= deterministic.rootVisits[1] ? 0 : 1;
    assert(deterministic.selectedAction == highest);
}

template <typename Mutation> void expectInvalidInference(Mutation mutation) {
    FixtureEvaluator evaluator;
    mutation(evaluator);
    bool rejected = false;
    try {
        (void) az::v2::search::FixedPuctSearch<FixtureState>::run(FixtureState{}, evaluator,
                                                                  terminalValue, configuration());
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
}

void testInvalidInferenceOutputs() {
    expectInvalidInference([](FixtureEvaluator &evaluator) { evaluator.rootPolicy = {0.5, 0.5}; });
    expectInvalidInference([](FixtureEvaluator &evaluator) { evaluator.request_id_offset = 1; });
    expectInvalidInference(
        [](FixtureEvaluator &evaluator) { evaluator.rootPolicy = {0.5, -0.1, 0.6}; });
    expectInvalidInference([](FixtureEvaluator &evaluator) {
        evaluator.rootPolicy = {0.5, std::numeric_limits<double>::quiet_NaN(), 0.5};
    });
    expectInvalidInference([](FixtureEvaluator &evaluator) {
        evaluator.rootPolicy = {std::numeric_limits<double>::max(),
                                std::numeric_limits<double>::max(), 0.0};
    });
    expectInvalidInference([](FixtureEvaluator &evaluator) {
        evaluator.rootValue = std::numeric_limits<double>::infinity();
    });
    expectInvalidInference([](FixtureEvaluator &evaluator) { evaluator.rootValue = 1.1; });
}

void testRootNoiseIsSeeded() {
    FixtureEvaluator firstEvaluator;
    FixtureEvaluator secondEvaluator;
    auto noisy = configuration(8);
    noisy.rootNoise.enabled = true;
    noisy.rootNoise.fraction = 0.5;
    const auto first = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, firstEvaluator, terminalValue, noisy);
    const auto second = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, secondEvaluator, terminalValue, noisy);
    assert(first.rootChildren[0].prior == second.rootChildren[0].prior);
    assert(first.rootVisits == second.rootVisits);

    FixtureEvaluator differentEvaluator;
    noisy.rootNoiseSeed += 1;
    const auto different = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, differentEvaluator, terminalValue, noisy);
    assert(first.rootChildren[0].prior != different.rootChildren[0].prior);
}

void testRandomPurposesUseIndependentStreams() {
    auto baselineConfiguration = configuration(16, 1.0, 1.0, 1);
    baselineConfiguration.rootNoise.enabled = true;
    baselineConfiguration.rootNoise.fraction = 0.5;
    FixtureEvaluator baselineEvaluator;
    const auto baseline = az::v2::search::FixedPuctSearch<FixtureState>::run(
        FixtureState{}, baselineEvaluator, terminalValue, baselineConfiguration);

    bool observedDifferentSample = false;
    for (uint64 actionSeed = 2; actionSeed < 64; ++actionSeed) {
        auto changedConfiguration = baselineConfiguration;
        changedConfiguration.actionSamplingSeed = actionSeed;
        FixtureEvaluator evaluator;
        const auto changed = az::v2::search::FixedPuctSearch<FixtureState>::run(
            FixtureState{}, evaluator, terminalValue, changedConfiguration);
        assert(changed.rootVisits == baseline.rootVisits);
        assert(changed.rootChildren[0].prior == baseline.rootChildren[0].prior);
        observedDifferentSample =
            observedDifferentSample || changed.selectedAction != baseline.selectedAction;
    }
    assert(observedDifferentSample);
}

template <typename Operation> void expectInvalidBatch(Operation operation) {
    bool rejected = false;
    try {
        operation();
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
}

void testInvalidConfigurationAndBatchContract() {
    FixtureEvaluator evaluator;
    auto invalid = configuration();
    invalid.treeReuse = true;
    bool rejected = false;
    try {
        (void) az::v2::search::FixedPuctSearch<FixtureState>::run(FixtureState{}, evaluator,
                                                                  terminalValue, invalid);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);

    for (const double invalidTemperature :
         {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        auto invalidFloat = configuration();
        invalidFloat.actionTemperature = invalidTemperature;
        rejected = false;
        try {
            (void) az::v2::search::FixedPuctSearch<FixtureState>::run(FixtureState{}, evaluator,
                                                                      terminalValue, invalidFloat);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        assert(rejected);
    }

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    for (const double invalidValue : {nan, infinity}) {
        auto invalidFloat = configuration();
        invalidFloat.explorationConstant = invalidValue;
        expectInvalidBatch([&invalidFloat] { invalidFloat.validate(); });

        invalidFloat = configuration();
        invalidFloat.backupDiscount = invalidValue;
        expectInvalidBatch([&invalidFloat] { invalidFloat.validate(); });

        invalidFloat = configuration();
        invalidFloat.noVisitedChildValue = invalidValue;
        expectInvalidBatch([&invalidFloat] { invalidFloat.validate(); });

        invalidFloat = configuration();
        invalidFloat.rootNoise.alpha = invalidValue;
        expectInvalidBatch([&invalidFloat] { invalidFloat.validate(); });

        invalidFloat = configuration();
        invalidFloat.rootNoise.fraction = invalidValue;
        expectInvalidBatch([&invalidFloat] { invalidFloat.validate(); });
    }
    auto zeroDiscount = configuration();
    zeroDiscount.backupDiscount = 0.0;
    expectInvalidBatch([&zeroDiscount] { zeroDiscount.validate(); });

    az::v2::search::SeededRandom random(1);
    expectInvalidBatch([&random, infinity] { (void) random.dirichlet(2, infinity); });

    FixtureEvaluator wideEvaluator;
    bool rejectedWideActionCount = false;
    try {
        (void) az::v2::search::FixedPuctSearch<WideActionState>::run(
            WideActionState{}, wideEvaluator,
            [](const WideActionState &) -> std::optional<double> { return 0.0; }, configuration());
    } catch (const std::logic_error &) {
        rejectedWideActionCount = true;
    }
    assert(rejectedWideActionCount);
}

void testBatchValidationAndOrderedAssociation() {
    using az::v2::inference::InferenceBatch;
    using az::v2::inference::InferenceBatchResult;
    const InferenceBatch<int32> batch{
        .requests =
            {
                {.requestId = 4, .encoding = 7, .actionCount = 3},
                {.requestId = 5, .encoding = 8, .actionCount = 2},
            },
    };
    const InferenceBatchResult validResult{
        .results =
            {
                {.requestId = 4, .policy = {0.2, 0.3, 0.5}, .value = 0.25},
                {.requestId = 5, .policy = {0.6, 0.4}, .value = -0.5},
            },
    };
    az::v2::inference::validateBatchResult(batch, validResult);

    expectInvalidBatch(
        [] { az::v2::inference::validateBatch(InferenceBatch<int32>{.requests = {}}); });
    expectInvalidBatch([] {
        az::v2::inference::validateBatch(
            InferenceBatch<int32>{.requests = {{.requestId = 1, .encoding = 0, .actionCount = 0}}});
    });
    expectInvalidBatch([] {
        az::v2::inference::validateBatch(
            InferenceBatch<int32>{.requests = {
                                      {.requestId = 1, .encoding = 0, .actionCount = 2},
                                      {.requestId = 1, .encoding = 1, .actionCount = 2},
                                  }});
    });
    expectInvalidBatch([&batch] {
        const InferenceBatchResult missing{
            .results = {{.requestId = 4, .policy = {0.2, 0.3, 0.5}, .value = 0.25}}};
        az::v2::inference::validateBatchResult(batch, missing);
    });
    expectInvalidBatch([&batch, &validResult] {
        InferenceBatchResult mismatched = validResult;
        mismatched.results[1].requestId = 6;
        az::v2::inference::validateBatchResult(batch, mismatched);
    });
    expectInvalidBatch([&batch, &validResult] {
        InferenceBatchResult invalid = validResult;
        invalid.results[0].policy[0] = -1.0;
        az::v2::inference::validateBatchResult(batch, invalid);
    });
}

} // namespace

int main() {
    testMaskingBackupAndExactAccounting();
    testZeroLegalMassBecomesUniform();
    testVisitedChildMeanFpu();
    testParentValueFpuUsesExpansionValueBeforeFirstBackup();
    testReducedParentFpuUsesVisitedPolicyMass();
    testAdaptiveStoppingChecksOnlyConfiguredIntervals();
    testBudgetMetadataDoesNotChangeSearch();
    testAdaptiveCadenceAtCapReportsFullBudget();
    testTerminalLeafSkipsInference();
    testCensoredTerminalLeafUsesInferenceWithoutExpansion();
    testTerminalRootIsTypedAndDoesNotInfer();
    testCensoredTerminalRootPreservesUnknownValue();
    testTemperatureAndRepeatability();
    testInvalidInferenceOutputs();
    testRootNoiseIsSeeded();
    testRandomPurposesUseIndependentStreams();
    testInvalidConfigurationAndBatchContract();
    testBatchValidationAndOrderedAssociation();
}
