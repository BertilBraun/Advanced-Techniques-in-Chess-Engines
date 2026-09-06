#include "TestRunner.hpp"
#include "games/chess/presentation/ChessSearchPresentation.hpp"
#include "position.h"
#include "search/SelfPlay.hpp"

#include <array>
#include <cmath>
#include <limits>

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
                                      const float loss, const bool validOutput = true,
                                      const bool legacyThreeTensorOutput = false) {
    torch::jit::script::Module model("batched_search_test");
    model.register_parameter("outcome_parameter",
                             validOutput ? torch::tensor({win, draw}) : torch::tensor({win}),
                             false);
    model.register_buffer("outcome_buffer", torch::tensor({loss}));
    if (legacyThreeTensorOutput) {
        model.define(R"JIT(
            def forward(self, boards):
                batch_size = boards.size(0)
                policies = torch.zeros((batch_size, )JIT" +
                     std::to_string(ChessEncoding::actionCount) + R"JIT(), device=boards.device)
                outcome = torch.cat((self.outcome_parameter, self.outcome_buffer))
                outcomes = outcome.unsqueeze(0).repeat((batch_size, 1))
                legacy = torch.zeros((batch_size, 8), device=boards.device)
                return policies, outcomes, legacy
        )JIT");
    } else {
        model.define(R"JIT(
            def forward(self, boards):
                batch_size = boards.size(0)
                policies = torch.zeros((batch_size, )JIT" +
                     std::to_string(ChessEncoding::actionCount) + R"JIT(), device=boards.device)
                outcome = torch.cat((self.outcome_parameter, self.outcome_buffer))
                outcomes = outcome.unsqueeze(0).repeat((batch_size, 1))
                return policies, outcomes
        )JIT");
    }
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
                                             std::vector<std::uint32_t> checkpoints = {}) {
    return {
        .root = search.newRoot(Board{}),
        .assigned_additional_visits = std::nullopt,
        .policy_checkpoint_visits = std::move(checkpoints),
        .parallel_searches = std::nullopt,
        .add_root_noise = false,
        .force_root_playouts = false,
        .checkpoint_detail = SearchCheckpointDetail::Policies,
        .root_ply = 0,
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
    const std::filesystem::path updatedModelPath = createTestModel("updated", 0.8F, 0.15F, 0.05F);
    const std::filesystem::path invalidModelPath =
        createTestModel("invalid", 0.5F, 0.0F, 0.5F, false);
    const std::filesystem::path legacyModelPath =
        createTestModel("legacy", 0.5F, 0.25F, 0.25F, true, true);
    const auto cleanup = [&]() {
        std::filesystem::remove(modelPath);
        std::filesystem::remove(updatedModelPath);
        std::filesystem::remove(invalidModelPath);
        std::filesystem::remove(legacyModelPath);
    };
    try {
        const std::array<std::uint32_t, 5> budgets = {100, 300, 600, 1'600, 2'400};
        const std::array<std::uint32_t, 5> expectedParallelism = {2, 2, 4, 8, 16};
        for (std::size_t index = 0; index < budgets.size(); ++index) {
            require(searchParallelism(budgets[index]) == expectedParallelism[index],
                    "search parallelism schedule changed");
        }

        const InferenceConfiguration runtimeParameters(0, modelPath.string(), InferenceDevice::Cpu);
        const SelfPlaySearchParameters searchParameters(16, treeSearchParameters(), 0.3F, 0.0F);
        const BatchedInferenceParameters inferenceParameters(2, 8, 1);
        ChessSelfPlaySearch search(runtimeParameters, searchParameters, inferenceParameters, 7);

        // A production self-play search runs flat to its baseline, bit-identical to an
        // additional-visit search of the same size.
        const auto productionResult = search.search({productionRequest(search)}).results.front();
        const auto flatResult = search.search({fixedRequest(search, 16, {}, 2)}).results.front();
        require(productionResult.final_visits == 16 && productionResult.starting_visits == 0,
                "a production search did not run flat to the baseline");
        require(productionResult.stop_reason == SearchStopReason::AdditionalVisits &&
                    flatResult.stop_reason == SearchStopReason::AdditionalVisits,
                "a production search did not report the flat stop reason");
        require(productionResult.final_visits == flatResult.final_visits &&
                    productionResult.root_value == flatResult.root_value,
                "a production search diverged from the flat baseline search");

        // Checkpoint exactness under parallelism, retained-root growth and validation.
        std::vector<ChessSelfPlaySearchRequest> heterogeneous;
        for (const std::uint32_t budget : budgets) {
            heterogeneous.push_back(fixedRequest(search, budget));
        }
        const auto heterogeneousResults = search.search(heterogeneous);
        for (std::size_t index = 0; index < budgets.size(); ++index) {
            require(heterogeneousResults.results[index].final_visits == budgets[index] &&
                        heterogeneousResults.results[index].parallel_searches ==
                            expectedParallelism[index],
                    "simultaneous heterogeneous search lost its per-request budget");
        }
        const auto checkpointResult =
            search.search({fixedRequest(search, 80, {20, 40, 80}, 4)}).results.front();
        require(checkpointResult.checkpoints.size() == 3 &&
                    checkpointResult.checkpoints[0].visits == 20 &&
                    checkpointResult.checkpoints[1].visits == 40 &&
                    checkpointResult.checkpoints[2].visits == 80,
                "continued search did not return every requested policy checkpoint exactly");
        require(std::ranges::all_of(checkpointResult.checkpoints,
                                    [](const SearchCheckpoint &checkpoint) {
                                        return !checkpoint.policy_target_visits.empty() &&
                                               std::isfinite(checkpoint.root_value);
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

        // Retained root: a production search on a warm root adds its baseline on top of the
        // visits already in the tree rather than restarting the count.
        ChessSelfPlaySearchRequest retained{
            .root = checkpointResult.root,
            .assigned_additional_visits = std::nullopt,
            .policy_checkpoint_visits = {},
            .parallel_searches = std::nullopt,
            .add_root_noise = false,
            .force_root_playouts = false,
            .checkpoint_detail = SearchCheckpointDetail::Policies,
            .root_ply = 12,
        };
        const auto warmResult = search.search({retained}).results.front();
        require(warmResult.starting_visits == 80 && warmResult.final_visits == 96,
                "a warm production search did not add its baseline in additional visits");

        search.refreshModel(8, updatedModelPath.string());
        require(search.modelGeneration() == 8, "refresh did not publish its model generation");
        try {
            search.refreshModel(9, invalidModelPath.string());
            throw std::runtime_error("invalid model refresh unexpectedly succeeded");
        } catch (const std::invalid_argument &) {
        }
        try {
            search.refreshModel(9, legacyModelPath.string());
            throw std::runtime_error("a legacy three-tensor model unexpectedly validated");
        } catch (const std::invalid_argument &) {
        }
        require(search.modelGeneration() == 8,
                "failed refresh published an unvalidated model generation");
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
    return 0;
}
