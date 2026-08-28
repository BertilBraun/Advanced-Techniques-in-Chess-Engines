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
                                      const float loss, const float searchBudgetLogit = 0.0F,
                                      const bool validOutput = true) {
    torch::jit::script::Module model("batched_search_test");
    model.register_parameter("outcome_parameter",
                             validOutput ? torch::tensor({win, draw}) : torch::tensor({win}),
                             false);
    model.register_buffer("outcome_buffer", torch::tensor({loss}));
    model.register_buffer("search_budget_logit", torch::tensor({searchBudgetLogit}));
    model.define(R"JIT(
        def forward(self, boards):
            batch_size = boards.size(0)
            policies = torch.zeros((batch_size, )JIT" +
                 std::to_string(ChessEncoding::actionCount) + R"JIT(), device=boards.device)
            outcome = torch.cat((self.outcome_parameter, self.outcome_buffer))
            outcomes = outcome.unsqueeze(0).repeat((batch_size, 1))
            search_budgets = self.search_budget_logit.unsqueeze(0).repeat((batch_size, 1))
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
    try {
        const SearchBudgetCurve flatCurve;
        const SearchBudgetCurve liveCurve(
            {0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45});
        require(std::abs(std::accumulate(liveCurve.multipliers.begin(), liveCurve.multipliers.end(),
                                         0.0) -
                         10.0) < 1e-12,
                "search-budget curve does not have uniform mean one");
        require(searchBudgetMultiplier(liveCurve, 0.0F) == 0.55 &&
                    searchBudgetMultiplier(liveCurve, std::nextafter(0.1F, 0.0F)) == 0.55 &&
                    searchBudgetMultiplier(liveCurve, 0.1F) == 0.65 &&
                    searchBudgetMultiplier(liveCurve, 1.0F) == 1.45,
                "search-budget curve lost its equal-width bucket boundaries");
        try {
            static_cast<void>(
                SearchBudgetCurve({1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1}));
            throw std::runtime_error("non-monotone search-budget curve unexpectedly validated");
        } catch (const std::invalid_argument &) {
        }
        try {
            static_cast<void>(
                SearchBudgetCurve({0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}));
            throw std::runtime_error("non-unit-mean search-budget curve unexpectedly validated");
        } catch (const std::invalid_argument &) {
        }
        try {
            static_cast<void>(
                SearchBudgetCurve({0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0}));
            throw std::runtime_error("nonpositive search-budget curve unexpectedly validated");
        } catch (const std::invalid_argument &) {
        }

        const std::array<std::uint32_t, 5> budgets = {100, 300, 600, 1'600, 2'400};
        const std::array<std::uint32_t, 5> expectedParallelism = {2, 2, 4, 8, 16};
        for (const std::size_t index : range(budgets.size())) {
            require(searchParallelism(budgets[index]) == expectedParallelism[index],
                    "production parallelism mapping changed");
        }
        require(searchParallelism(100'000) == 16, "production parallelism exceeded its cap");

        SearchBudgetAllocator allocator;
        const PredictedSearchBudgetLimit allocationLimit(101, liveCurve);
        std::uint64_t assignedTotal = 0;
        constexpr std::uint32_t allocationCount = 10'000;
        for (const auto index : range(allocationCount)) {
            static_cast<void>(index);
            assignedTotal += allocator.assign(allocationLimit, 0.0F);
        }
        const auto expectedFloorError = static_cast<std::int64_t>(assignedTotal) -
                                        static_cast<std::int64_t>(allocationCount) * 101;
        const std::int64_t strictErrorBound =
            std::max<std::int64_t>(101 - static_cast<std::int64_t>(std::ceil(101.0 * 0.55)),
                                   static_cast<std::int64_t>(std::floor(101.0 * 1.45)) - 101) +
            1;
        require(allocator.spendError() == expectedFloorError &&
                    std::abs(allocator.spendError()) < strictErrorBound,
                "all-floor predictions violated cumulative mean-spend accounting");
        const std::int64_t errorBeforeFlatCurve = allocator.spendError();
        const PredictedSearchBudgetLimit flatLimit(101, flatCurve);
        std::uint64_t flatAssignedTotal = 0;
        for (const auto index : range(100)) {
            static_cast<void>(index);
            flatAssignedTotal += allocator.assign(flatLimit, 0.0F);
        }
        const std::int64_t flatCorrection =
            static_cast<std::int64_t>(flatAssignedTotal) - 100 * 101;
        require(allocator.spendError() == 0 && flatCorrection == -errorBeforeFlatCurve,
                "flat curve did not repay the existing cumulative spend error");
        for (const auto index : range(allocationCount)) {
            static_cast<void>(index);
            static_cast<void>(allocator.assign(allocationLimit, 1.0F));
        }
        require(std::abs(allocator.spendError()) < strictErrorBound,
                "all-ceiling predictions violated cumulative mean-spend accounting");
        const SearchBudgetCurve cappedCurve({0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 9.1});
        SearchBudgetAllocator cappedAllocator;
        require(cappedAllocator.assign(PredictedSearchBudgetLimit(101, cappedCurve), 1.0F) == 808,
                "production allocation exceeded the eight-times-baseline deep reference");
        require(maximumAdditionalVisits(PredictedSearchBudgetLimit(101, cappedCurve)) == 808,
                "predicted search capacity did not use the eight-times-baseline cap");

        const InferenceConfiguration runtimeParameters(0, modelPath.string(), InferenceDevice::Cpu);
        const SelfPlaySearchParameters searchParameters(16, flatCurve, treeSearchParameters(), 0.3F,
                                                        0.0F);
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
            require(result.search_budget_logit == 0.0F &&
                        std::abs(result.predicted_search_budget - 0.5F) < 1e-7F,
                    "native root did not preserve raw and bounded search-budget outputs");
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

        const SelfPlaySearchParameters liveCurveParameters(101, liveCurve, treeSearchParameters(),
                                                           0.3F, 0.0F);
        ChessSelfPlaySearch liveCurveSearch(runtimeParameters, liveCurveParameters,
                                            inferenceParameters);
        const auto liveCurveResult =
            liveCurveSearch.search({productionRequest(liveCurveSearch)}).results.front();
        require(liveCurveResult.spend_residual == liveCurveSearch.spendResidual(),
                "predicted request did not expose its exact post-assignment spend residual");
        const std::int64_t residualBeforeExplicit = liveCurveSearch.spendResidual();
        const auto explicitResult =
            liveCurveSearch.search({fixedRequest(liveCurveSearch, 50, {}, 1)}).results.front();
        require(explicitResult.spend_residual == residualBeforeExplicit &&
                    liveCurveSearch.spendResidual() == residualBeforeExplicit,
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
        require(std::abs(updated.search_budget_logit - 2.0F) < 1e-7F &&
                    std::abs(updated.predicted_search_budget - 0.880797F) < 1e-6F,
                "model refresh did not update the raw and bounded budget prediction");
        try {
            search.refreshModel(9, invalidModelPath.string());
            throw std::runtime_error("invalid model refresh unexpectedly succeeded");
        } catch (const std::invalid_argument &) {
        }
        require(search.modelGeneration() == 8,
                "failed refresh published an unvalidated model generation");
    } catch (...) {
        std::filesystem::remove(modelPath);
        std::filesystem::remove(updatedModelPath);
        std::filesystem::remove(invalidModelPath);
        throw;
    }
    std::filesystem::remove(modelPath);
    std::filesystem::remove(updatedModelPath);
    std::filesystem::remove(invalidModelPath);
    return 0;
}
