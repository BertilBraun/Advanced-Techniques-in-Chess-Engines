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
        constexpr std::array<std::uint32_t, 10> widths = {1'186, 384, 268, 210, 140,
                                                          159,   200, 152, 107, 194};
        constexpr std::array<std::uint32_t, 10> numerators = {750,   1'500, 2'250, 3'000,  3'750,
                                                              4'500, 6'000, 9'000, 12'000, 18'000};
        std::uint64_t weightedNumerator = 0;
        for (const std::size_t index : range(widths.size())) {
            weightedNumerator += static_cast<std::uint64_t>(widths[index]) * numerators[index];
        }
        require(weightedNumerator == 3'000ULL * 3'761ULL,
                "search-budget curve does not have exact uniform mean one");
        require(std::abs(searchBudgetMultiplier(0.0F) - 750.0F / 3'761.0F) < 1e-7F &&
                    std::abs(searchBudgetMultiplier(1.0F) - 18'000.0F / 3'761.0F) < 1e-7F,
                "search-budget curve lost its exact floor or ceiling");
        require(searchBudgetMultiplier(1'186.0F / 3'000.0F) == 1'500.0F / 3'761.0F,
                "search-budget boundary did not select the right-open interval");

        const std::array<std::uint32_t, 5> budgets = {100, 300, 600, 1'600, 2'400};
        const std::array<std::uint32_t, 5> expectedParallelism = {1, 2, 4, 8, 16};
        for (const std::size_t index : range(budgets.size())) {
            require(searchParallelism(budgets[index]) == expectedParallelism[index],
                    "production parallelism mapping changed");
        }
        require(searchParallelism(100'000) == 16, "production parallelism exceeded its cap");

        SearchBudgetAllocator allocator;
        const PredictedSearchBudgetLimit allocationLimit(101, 1.0F);
        std::uint64_t assignedTotal = 0;
        constexpr std::uint32_t allocationCount = 10'000;
        for (const auto index : range(allocationCount)) {
            static_cast<void>(index);
            assignedTotal += allocator.assign(allocationLimit, 0.0F);
        }
        const auto expectedFloorError = static_cast<std::int64_t>(assignedTotal) -
                                        static_cast<std::int64_t>(allocationCount) * 101;
        const std::int64_t strictErrorBound =
            std::max<std::int64_t>(
                101 - static_cast<std::int64_t>(std::ceil(101.0 * 750.0 / 3'761.0)),
                static_cast<std::int64_t>(std::floor(101.0 * 18'000.0 / 3'761.0)) - 101) +
            1;
        require(allocator.spendError() == expectedFloorError &&
                    std::abs(allocator.spendError()) < strictErrorBound,
                "all-floor predictions violated cumulative mean-spend accounting");
        const std::int64_t errorBeforeFlatBlend = allocator.spendError();
        const PredictedSearchBudgetLimit flatLimit(101, 0.0F);
        for (const auto index : range(100)) {
            static_cast<void>(index);
            require(allocator.assign(flatLimit, 0.0F) == 101,
                    "zero blend did not assign the exact baseline");
        }
        require(allocator.spendError() == errorBeforeFlatBlend,
                "zero blend worsened an existing cumulative spend error");
        for (const auto index : range(allocationCount)) {
            static_cast<void>(index);
            static_cast<void>(allocator.assign(allocationLimit, 1.0F));
        }
        require(std::abs(allocator.spendError()) < strictErrorBound,
                "all-ceiling predictions violated cumulative mean-spend accounting");

        const InferenceConfiguration runtimeParameters(0, modelPath.string(), InferenceDevice::Cpu);
        const SelfPlaySearchParameters searchParameters(16, 0.0F, treeSearchParameters(), 0.3F,
                                                        0.0F);
        const BatchedInferenceParameters inferenceParameters(2, 8, 1);
        ChessSelfPlaySearch search(runtimeParameters, searchParameters, inferenceParameters, 7);
        std::vector<ChessSelfPlaySearchRequest> productionRequests = {productionRequest(search),
                                                                      productionRequest(search)};
        const auto production = search.search(productionRequests, true);
        require(production.simulations_completed == 32 && production.results.size() == 2,
                "flat predicted allocation completed the wrong total visits");
        for (const auto &result : production.results) {
            require(result.assigned_additional_visits == 16 && result.starting_visits == 0 &&
                        result.final_visits == 16,
                    "predicted allocation did not expose additional-visit semantics");
            require(result.parallel_searches == 1,
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

        const SelfPlaySearchParameters adaptiveParameters(101, 1.0F, treeSearchParameters(), 0.3F,
                                                          0.0F);
        ChessSelfPlaySearch adaptiveSearch(runtimeParameters, adaptiveParameters,
                                           inferenceParameters);
        const auto adaptiveResult =
            adaptiveSearch.search({productionRequest(adaptiveSearch, 600)}).results.front();
        require(adaptiveResult.spend_residual == adaptiveSearch.spendResidual(),
                "predicted request did not expose its exact post-assignment spend residual");
        const std::int64_t residualBeforeExplicit = adaptiveSearch.spendResidual();
        const auto explicitResult =
            adaptiveSearch.search({fixedRequest(adaptiveSearch, 50, {}, 1)}).results.front();
        require(explicitResult.spend_residual == residualBeforeExplicit &&
                    adaptiveSearch.spendResidual() == residualBeforeExplicit,
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
