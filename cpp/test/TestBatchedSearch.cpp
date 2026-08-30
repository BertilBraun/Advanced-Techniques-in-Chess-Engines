#include "TestRunner.hpp"
#include "games/chess/presentation/ChessSearchPresentation.hpp"
#include "position.h"
#include "search/SelfPlay.hpp"

#include <array>

namespace {
using ChessSelfPlaySearch = GameSelfPlaySearch<ChessGame>;
using ChessSelfPlaySearchRequest = SelfPlaySearchRequest<ChessGame>;

TreeSearchParameters treeSearchParameters(const float explorationConstant = 1.5F,
                                          const float valueDiscountPerPly = 1.0F) {
    return TreeSearchParameters(explorationConstant,
                                FirstPlayUrgencyParameters(FirstPlayUrgencyKind::Zero), 0.0F,
                                valueDiscountPerPly);
}

std::filesystem::path createTestModel(const std::string &name, const float win, const float draw,
                                      const float loss, const float searchBudgetValue = 0.0F,
                                      const bool validOutput = true,
                                      const bool validBudgetWidth = true) {
    torch::jit::script::Module model("batched_search_test");
    model.register_parameter("outcome_parameter",
                             validOutput ? torch::tensor({win, draw}) : torch::tensor({win}),
                             false);
    model.register_buffer("outcome_buffer", torch::tensor({loss}));
    model.register_buffer("search_budget_curve",
                          validBudgetWidth
                              ? torch::full({static_cast<std::int64_t>(SEARCH_BUDGET_CURVE_POINTS)},
                                            searchBudgetValue)
                              : torch::full({1}, searchBudgetValue));
    model.define(R"JIT(
        def forward(self, boards):
            batch_size = boards.size(0)
            policies = torch.zeros((batch_size, )JIT" +
                 std::to_string(ChessEncoding::actionCount) + R"JIT(), device=boards.device)
            outcome = torch.cat((self.outcome_parameter, self.outcome_buffer))
            outcomes = outcome.unsqueeze(0).repeat((batch_size, 1))
            search_budgets = self.search_budget_curve.unsqueeze(0).repeat((batch_size, 1))
            return policies, outcomes, search_budgets
    )JIT");
    const auto uniqueSuffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("batched-search-test-" + name + "-" + std::to_string(uniqueSuffix) + ".jit.pt");
    model.save(path.string());
    return path;
}

void require(const bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

ChessSelfPlaySearchRequest productionRequest(ChessSelfPlaySearch &search,
                                             const std::size_t maximumCapacity = 0) {
    return {
        .root = search.newRoot(Board{}, maximumCapacity),
        .assigned_additional_visits = std::nullopt,
        .policy_checkpoint_visits = {},
        .parallel_searches = std::nullopt,
        .add_root_noise = false,
        .force_root_playouts = false,
    };
}

ChessSelfPlaySearchRequest
fixedRequest(ChessSelfPlaySearch &search, const std::uint32_t additionalVisits,
             std::vector<std::uint32_t> checkpoints = {},
             const std::optional<std::uint32_t> parallelSearches = std::nullopt) {
    return {
        .root = search.newRoot(
            Board{}, std::max<std::uint32_t>(search.arenaCapacity(), additionalVisits + 32U)),
        .assigned_additional_visits = additionalVisits,
        .policy_checkpoint_visits = std::move(checkpoints),
        .parallel_searches = parallelSearches,
        .add_root_noise = false,
        .force_root_playouts = false,
        .checkpoint_detail = SearchCheckpointDetail::Policies,
    };
}

} // namespace

int runBatchedSearchTests() {
    Stockfish::Bitboards::init();
    Stockfish::Position::init();
    const std::filesystem::path modelPath =
        createTestModel("initial", 1.0F / 3.0F, 1.0F / 3.0F, 1.0F / 3.0F);
    const std::filesystem::path updatedModelPath =
        createTestModel("updated", 0.8F, 0.15F, 0.05F, 2.0F);
    const std::filesystem::path invalidModelPath =
        createTestModel("invalid", 0.5F, 0.5F, 0.0F, 0.0F, false);
    const std::filesystem::path narrowBudgetModelPath =
        createTestModel("narrow-budget", 0.5F, 0.5F, 0.0F, 0.0F, true, false);
    try {
        const SearchBudgetPolicy flatPolicy;
        require(!flatPolicy.apply_learned, "default search-budget policy is not flat");
        const std::array<double, 10> gridMultiples = {0.125, 0.2, 1.0 / 3.0, 0.5, 2.0 / 3.0,
                                                      1.0,   1.5, 2.0,       3.0, 4.0};
        const std::array<double, 10> unitSigma = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        const SearchBudgetPolicy learnedPolicy(gridMultiples, unitSigma, 1.0, 0.8, true);
        try {
            std::array<double, 10> nonIncreasing = gridMultiples;
            nonIncreasing[3] = nonIncreasing[2];
            static_cast<void>(SearchBudgetPolicy(nonIncreasing, unitSigma, 0.0, 0.8, true));
            throw std::runtime_error("non-increasing grid multiples unexpectedly validated");
        } catch (const std::invalid_argument &) {
        }
        try {
            std::array<double, 10> withoutBaseline = gridMultiples;
            withoutBaseline[5] = 1.1;
            static_cast<void>(SearchBudgetPolicy(withoutBaseline, unitSigma, 0.0, 0.8, true));
            throw std::runtime_error("grid without the flat multiple unexpectedly validated");
        } catch (const std::invalid_argument &) {
        }
        try {
            std::array<double, 10> zeroSigma = unitSigma;
            zeroSigma[0] = 0.0;
            static_cast<void>(SearchBudgetPolicy(gridMultiples, zeroSigma, 0.0, 0.8, true));
            throw std::runtime_error("nonpositive sigma unexpectedly validated");
        } catch (const std::invalid_argument &) {
        }
        try {
            static_cast<void>(SearchBudgetPolicy(gridMultiples, unitSigma, 0.0, 1.0, true));
            throw std::runtime_error("unit selection threshold unexpectedly validated");
        } catch (const std::invalid_argument &) {
        }

        const SearchBudgetCurvePrediction unsortedPrediction = {5.0F, 1.0F, 4.0F, 2.0F, 3.0F,
                                                                2.0F, 9.0F, 0.0F, 8.0F, 7.0F};
        const SearchBudgetCurvePrediction expectedProjection = {5.0F, 1.0F, 1.0F, 1.0F, 1.0F,
                                                                1.0F, 1.0F, 0.0F, 0.0F, 0.0F};
        require(projectNonIncreasing(unsortedPrediction) == expectedProjection,
                "isotonic projection is not the running minimum from the cheapest budget upward");

        // A well-formed decreasing curve must survive the projection unchanged; a suffix minimum
        // would flatten it to its deepest value and reduce selection to a two-point rule.
        const SearchBudgetCurvePrediction decreasingPrediction = {-1.0F, -1.4F, -1.9F, -2.3F, -2.6F,
                                                                  -3.0F, -3.4F, -3.9F, -4.5F, -5.2F};
        require(projectNonIncreasing(decreasingPrediction) == decreasingPrediction,
                "a well-formed decreasing curve was not a fixed point of the projection");

        // With log_tau = 1 and unit sigma, Phi(1) ~= 0.841 > 0.8 qualifies a zero prediction.
        SearchBudgetCurvePrediction cheapPrediction{};
        require(selectBudgetIndex(learnedPolicy, cheapPrediction) == 0,
                "a confidently cheap curve did not select the cheapest grid point");
        SearchBudgetCurvePrediction hardPrediction;
        hardPrediction.fill(5.0F);
        require(selectBudgetIndex(learnedPolicy, hardPrediction) == 9,
                "an unqualified curve did not fall back to the deepest grid point");
        SearchBudgetCurvePrediction dippingPrediction;
        dippingPrediction.fill(5.0F);
        dippingPrediction[6] = -3.0F;
        // Isotonic projection pulls indices 0..6 down to -3, so the earliest index qualifies.
        require(selectBudgetIndex(learnedPolicy, dippingPrediction) == 0,
                "isotonic projection did not propagate a deep-budget dip to cheaper budgets");
        // A constant projected curve leaves only sigma to separate grid points: wide sigma
        // blocks the cheap points and the first tight point qualifies.
        const std::array<double, 10> mixedSigma = {5.0, 5.0, 5.0, 5.0, 1.0,
                                                   1.0, 1.0, 1.0, 1.0, 1.0};
        const SearchBudgetPolicy mixedSigmaPolicy(gridMultiples, mixedSigma, -2.0, 0.8, true);
        SearchBudgetCurvePrediction constantPrediction;
        constantPrediction.fill(-4.0F);
        require(selectBudgetIndex(mixedSigmaPolicy, constantPrediction) == 4,
                "selection did not choose the lowest qualifying grid point");

        const std::array<std::uint32_t, 5> budgets = {100, 300, 600, 1'600, 2'400};
        const std::array<std::uint32_t, 5> expectedParallelism = {2, 2, 4, 8, 16};
        for (const std::size_t index : range(budgets.size())) {
            require(searchParallelism(budgets[index]) == expectedParallelism[index],
                    "production parallelism mapping changed");
        }
        require(searchParallelism(100'000) == 16, "production parallelism exceeded its cap");

        SearchBudgetAllocator allocator;
        const PredictedSearchBudgetLimit learnedLimit(101, learnedPolicy);
        const AssignedSearchBudget firstAssigned = allocator.assign(learnedLimit, cheapPrediction);
        require(firstAssigned.selected_index == 0 && firstAssigned.additional_visits == 13,
                "first learned assignment did not round the cheapest grid budget");
        std::uint64_t assignedTotal = firstAssigned.additional_visits;
        constexpr std::uint32_t allocationCount = 10'000;
        for (const auto index : range(allocationCount - 1)) {
            static_cast<void>(index);
            assignedTotal += allocator.assign(learnedLimit, cheapPrediction).additional_visits;
        }
        require(std::abs(static_cast<std::int64_t>(assignedTotal) -
                         static_cast<std::int64_t>(allocationCount) * 101) <= 8 * 101 + 1,
                "constantly cheap predictions violated cumulative mean-spend accounting");
        require(allocator.spendError() == static_cast<std::int64_t>(assignedTotal) -
                                              static_cast<std::int64_t>(allocationCount) * 101,
                "spend ledger does not equal assigned-minus-baseline");
        const std::int64_t errorBeforeFlat = allocator.spendError();
        const PredictedSearchBudgetLimit flatLimit(101, flatPolicy);
        for (const auto index : range(100)) {
            static_cast<void>(index);
            const AssignedSearchBudget flatAssigned = allocator.assign(flatLimit, cheapPrediction);
            require(flatAssigned.additional_visits == 101 && flatAssigned.selected_index == -1,
                    "flat policy did not assign exactly the baseline");
        }
        require(allocator.spendError() == errorBeforeFlat,
                "flat policy mutated the learned spend ledger");
        require(maximumAdditionalVisits(learnedLimit) == 808,
                "predicted search capacity did not use the eight-times-baseline cap");

        const InferenceConfiguration runtimeParameters(0, modelPath.string(), InferenceDevice::Cpu);
        const SelfPlaySearchParameters searchParameters(16, flatPolicy, treeSearchParameters(),
                                                        0.3F, 0.0F);
        const BatchedInferenceParameters inferenceParameters(2, 8, 1);
        ChessSelfPlaySearch search(runtimeParameters, searchParameters, inferenceParameters, 7);
        const auto baselineSizedRoot = search.newRoot(Board{});
        require(baselineSizedRoot.tree().capacity() == 19 &&
                    baselineSizedRoot.tree().maximumCapacity() == 145,
                "production root arena capacities were " +
                    std::to_string(baselineSizedRoot.tree().capacity()) + "/" +
                    std::to_string(baselineSizedRoot.tree().maximumCapacity()));
        const auto grownArenaResult = search.search({{
            .root = baselineSizedRoot,
            .assigned_additional_visits = 24,
            .policy_checkpoint_visits = {},
            .parallel_searches = 2,
            .add_root_noise = false,
            .force_root_playouts = false,
        }});
        require(grownArenaResult.results.front().root.tree().capacity() == 38 &&
                    grownArenaResult.results.front().root.tree().maximumCapacity() == 145,
                "production root did not grow within its configured adaptive-search bound");
        const auto predictionRoot = search.newRoot(Board{}, 3);
        require(predictionRoot.tree().maximumCapacity() == 3,
                "per-root capacity did not constrain the initial arena allocation");
        const auto tightDeepSearch = search.search({{
            .root = search.newRoot(Board{}, 19),
            .assigned_additional_visits = 16,
            .policy_checkpoint_visits = {},
            .parallel_searches = 2,
            .add_root_noise = false,
            .force_root_playouts = true,
        }});
        require(tightDeepSearch.results.front().final_visits == 16,
                "deep-label arena bound did not reserve parallel-search and reroot slots");
        std::vector<ChessSelfPlaySearchRequest> productionRequests = {productionRequest(search),
                                                                      productionRequest(search)};
        const auto production = search.search(productionRequests, true);
        require(production.simulations_completed == 32 && production.results.size() == 2,
                "flat predicted allocation completed the wrong total visits");
        for (const auto &result : production.results) {
            require(result.assigned_additional_visits == 16 && result.starting_visits == 0 &&
                        result.final_visits == 16,
                    "predicted allocation did not expose additional-visit semantics");
            require(result.parallel_searches == 2,
                    "small predicted budget used the wrong parallelism");
            require(result.selected_budget_index == -1,
                    "flat allocation reported a learned grid selection");
            require(std::ranges::all_of(result.predicted_budget_curve,
                                        [](const float value) { return value == 0.0F; }),
                    "native root did not preserve the predicted budget curve");
        }

        ChessSelfPlaySearchRequest retained{
            .root = production.results.front().root,
            .assigned_additional_visits = std::nullopt,
            .policy_checkpoint_visits = {},
            .parallel_searches = std::nullopt,
            .add_root_noise = false,
            .force_root_playouts = false,
        };
        const auto retainedResult = search.search({retained}).results.front();
        require(retainedResult.starting_visits == 16 &&
                    retainedResult.assigned_additional_visits == 16 &&
                    retainedResult.final_visits == 32,
                "retained root treated the assigned budget as an absolute visit limit");

        const SelfPlaySearchParameters learnedParameters(16, learnedPolicy, treeSearchParameters(),
                                                         0.3F, 0.0F);
        ChessSelfPlaySearch learnedSearch(runtimeParameters, learnedParameters,
                                          inferenceParameters);
        const auto learnedResult =
            learnedSearch.search({productionRequest(learnedSearch)}).results.front();
        // The test model predicts a zero curve, which qualifies the cheapest grid point.
        require(learnedResult.selected_budget_index == 0 &&
                    learnedResult.assigned_additional_visits == 2,
                "learned policy did not select and round the cheapest grid budget");
        require(learnedResult.spend_residual == learnedSearch.spendResidual(),
                "predicted request did not expose its exact post-assignment spend residual");
        const std::int64_t residualBeforeExplicit = learnedSearch.spendResidual();
        const auto explicitResult =
            learnedSearch.search({fixedRequest(learnedSearch, 50, {}, 1)}).results.front();
        require(explicitResult.spend_residual == residualBeforeExplicit &&
                    learnedSearch.spendResidual() == residualBeforeExplicit,
                "explicit deep-label budget mutated the production spend ledger");

        std::vector<ChessSelfPlaySearchRequest> heterogeneous;
        for (const std::uint32_t budget : budgets) {
            heterogeneous.push_back(fixedRequest(search, budget));
        }
        const auto heterogeneousResults = search.search(heterogeneous);
        for (const std::size_t index : range(budgets.size())) {
            require(heterogeneousResults.results[index].assigned_additional_visits ==
                            budgets[index] &&
                        heterogeneousResults.results[index].parallel_searches ==
                            expectedParallelism[index] &&
                        heterogeneousResults.results[index].final_visits == budgets[index],
                    "simultaneous heterogeneous search lost its per-request budget or parallelism");
        }

        const auto checkpointResult =
            search.search({fixedRequest(search, 80, {20, 40, 80}, 1)}).results.front();
        require(checkpointResult.checkpoints.size() == 3 &&
                    checkpointResult.checkpoints[0].visits == 20 &&
                    checkpointResult.checkpoints[1].visits == 40 &&
                    checkpointResult.checkpoints[2].visits == 80,
                "continued search did not return every requested policy checkpoint");
        require(std::ranges::all_of(checkpointResult.checkpoints,
                                    [](const SearchCheckpoint &checkpoint) {
                                        return !checkpoint.policy_target_visits.empty();
                                    }),
                "policy checkpoint detail omitted a requested policy snapshot");
        require(std::ranges::all_of(checkpointResult.checkpoints,
                                    [](const SearchCheckpoint &checkpoint) {
                                        return std::isfinite(checkpoint.root_value) &&
                                               checkpoint.root_value >= -1.0F &&
                                               checkpoint.root_value <= 1.0F;
                                    }),
                "policy checkpoints did not record a bounded root value");

        try {
            static_cast<void>(search.search({fixedRequest(search, 80, {40, 20}, 1)}));
            throw std::runtime_error("unsorted checkpoint request unexpectedly validated");
        } catch (const std::invalid_argument &) {
        }
        try {
            static_cast<void>(search.search({fixedRequest(search, 80, {20, 20}, 1)}));
            throw std::runtime_error("duplicate checkpoint request unexpectedly validated");
        } catch (const std::invalid_argument &) {
        }

        search.refreshModel(8, updatedModelPath.string());
        require(search.modelGeneration() == 8, "refresh did not publish its model generation");
        const auto updated = search.search({productionRequest(search)}).results.front();
        require(
            std::ranges::all_of(updated.predicted_budget_curve,
                                [](const float value) { return std::abs(value - 2.0F) < 1e-7F; }),
            "model refresh did not update the predicted budget curve");
        try {
            search.refreshModel(9, invalidModelPath.string());
            throw std::runtime_error("invalid model refresh unexpectedly succeeded");
        } catch (const std::invalid_argument &) {
        }
        try {
            search.refreshModel(9, narrowBudgetModelPath.string());
            throw std::runtime_error("narrow search-budget head unexpectedly validated");
        } catch (const std::invalid_argument &) {
        }
        require(search.modelGeneration() == 8,
                "failed refresh published an unvalidated model generation");
    } catch (...) {
        std::filesystem::remove(modelPath);
        std::filesystem::remove(updatedModelPath);
        std::filesystem::remove(invalidModelPath);
        std::filesystem::remove(narrowBudgetModelPath);
        throw;
    }
    std::filesystem::remove(modelPath);
    std::filesystem::remove(updatedModelPath);
    std::filesystem::remove(invalidModelPath);
    std::filesystem::remove(narrowBudgetModelPath);
    return 0;
}
