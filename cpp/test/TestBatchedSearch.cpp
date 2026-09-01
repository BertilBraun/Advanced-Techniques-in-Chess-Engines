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

std::filesystem::path createStopPredictorModel(const std::string &name, const float uncertainty) {
    torch::jit::script::Module model("stop_predictor_test");
    model.register_buffer("uncertainty", torch::tensor({uncertainty}));
    model.define(R"JIT(
        def forward(self, features):
            return self.uncertainty.unsqueeze(0).repeat((features.size(0), 1))
    )JIT");
    const auto uniqueSuffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("stop-predictor-test-" + name + "-" + std::to_string(uniqueSuffix) + ".jit.pt");
    model.save(path.string());
    return path;
}

void require(const bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// A published closed policy still carries the configured checkpoint multiples so audit searches
// can run to the cap and record labels while the gate is closed.
SearchStopPolicy closedPolicy() {
    return SearchStopPolicy({0.5, 1.0, 1.5}, {0.0, 0.0, 0.0}, 100.0, 2.0, nullptr, false);
}

SearchStopPolicy openPolicy(const std::filesystem::path &predictorPath,
                            std::vector<double> thresholds,
                            const double movementGuardEpsilon = 100.0,
                            std::vector<double> multiples = {0.5, 1.0, 1.5}) {
    return SearchStopPolicy(std::move(multiples), std::move(thresholds), movementGuardEpsilon, 2.0,
                            std::make_shared<SearchStopPredictor>(predictorPath.string()), true);
}

ChessSelfPlaySearchRequest productionRequest(ChessSelfPlaySearch &search,
                                             std::vector<std::uint32_t> checkpoints = {},
                                             const bool audit = false) {
    return {
        .root = search.newRoot(Board{}),
        .assigned_additional_visits = std::nullopt,
        .policy_checkpoint_visits = std::move(checkpoints),
        .parallel_searches = std::nullopt,
        .add_root_noise = false,
        .force_root_playouts = false,
        .checkpoint_detail = SearchCheckpointDetail::Policies,
        .root_ply = 0,
        .audit = audit,
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
    const std::filesystem::path certainPredictorPath = createStopPredictorModel("certain", 0.0F);
    const std::filesystem::path uncertainPredictorPath =
        createStopPredictorModel("uncertain", 1.0F);
    const auto cleanup = [&]() {
        std::filesystem::remove(modelPath);
        std::filesystem::remove(updatedModelPath);
        std::filesystem::remove(invalidModelPath);
        std::filesystem::remove(legacyModelPath);
        std::filesystem::remove(certainPredictorPath);
        std::filesystem::remove(uncertainPredictorPath);
    };
    try {
        {
            const SearchStopPolicy closed = closedPolicy();
            require(!closed.apply_learned, "the default stop policy is not closed");
            const StoppableSearchLimit closedLimit(16, closed);
            require(!closedLimit.searchesToCap() && closedLimit.capAdditionalVisits() == 16,
                    "a closed policy did not collapse the cap to the baseline");
            const StoppableSearchLimit auditLimit(16, closed, 0, true);
            require(auditLimit.searchesToCap() && auditLimit.capAdditionalVisits() == 32,
                    "an audit search under a closed policy did not search to the cap");
        }
        try {
            static_cast<void>(SearchStopPolicy({0.5, 0.5}, {0.1, 0.1}, 0.05, 2.0, nullptr, false));
            throw std::runtime_error("non-increasing checkpoint multiples validated");
        } catch (const std::invalid_argument &) {
        }
        try {
            static_cast<void>(SearchStopPolicy({0.5, 2.5}, {0.1, 0.1}, 0.05, 2.0, nullptr, false));
            throw std::runtime_error("a checkpoint above the cap validated");
        } catch (const std::invalid_argument &) {
        }
        try {
            static_cast<void>(SearchStopPolicy({0.5}, {0.1}, 0.05, 2.0, nullptr, true));
            throw std::runtime_error("an applied policy without a predictor validated");
        } catch (const std::invalid_argument &) {
        }

        const std::array<std::uint32_t, 5> budgets = {100, 300, 600, 1'600, 2'400};
        const std::array<std::uint32_t, 5> expectedParallelism = {2, 2, 4, 8, 16};
        for (std::size_t index = 0; index < budgets.size(); ++index) {
            require(searchParallelism(budgets[index]) == expectedParallelism[index],
                    "search parallelism schedule changed");
        }

        const InferenceConfiguration runtimeParameters(0, modelPath.string(), InferenceDevice::Cpu);
        const SelfPlaySearchParameters searchParameters(16, closedPolicy(), treeSearchParameters(),
                                                        0.3F, 0.0F);
        const BatchedInferenceParameters inferenceParameters(2, 8, 1);
        ChessSelfPlaySearch search(runtimeParameters, searchParameters, inferenceParameters, 7);

        // The fail-closed identity: a closed stoppable production search must be bit-identical
        // to a flat additional-visit search of the baseline.
        const auto closedResult = search.search({productionRequest(search)}).results.front();
        const auto flatResult = search.search({fixedRequest(search, 16, {}, 2)}).results.front();
        require(closedResult.final_visits == 16 && closedResult.starting_visits == 0,
                "closed policy did not run flat to the baseline");
        require(closedResult.stop_reason == SearchStopReason::AdditionalVisits &&
                    flatResult.stop_reason == SearchStopReason::AdditionalVisits,
                "closed policy did not report the flat stop reason");
        require(closedResult.stop_checkpoint_index == -1 && closedResult.checkpoints.empty() &&
                    closedResult.stop_features.empty(),
                "closed policy evaluated the stop rule");
        require(closedResult.final_visits == flatResult.final_visits &&
                    closedResult.root_value == flatResult.root_value,
                "closed stoppable search diverged from the flat baseline search");
        try {
            static_cast<void>(search.search({productionRequest(search, {8})}));
            throw std::runtime_error("closed policy accepted checkpoints");
        } catch (const std::invalid_argument &) {
        }

        // Shadow audit: runs to the cap, records verdicts at every checkpoint, never stops.
        const auto auditResult =
            search.search({productionRequest(search, {8, 16, 24}, true)}).results.front();
        require(auditResult.final_visits == 32 &&
                    auditResult.stop_reason == SearchStopReason::CapReached,
                "audit search did not run to the cap");
        require(auditResult.checkpoints.size() == 3 && auditResult.stop_features.size() == 3 &&
                    auditResult.guard_movements.size() == 3 &&
                    auditResult.stop_checkpoint_index == -1,
                "audit search did not record every checkpoint evaluation");
        require(std::ranges::all_of(auditResult.stop_probabilities,
                                    [](const double value) { return value == -1.0; }),
                "audit search under a closed policy evaluated a predictor");
        for (const StopPredictorFeatures &features : auditResult.stop_features) {
            require(std::ranges::all_of(features,
                                        [](const double value) { return std::isfinite(value); }),
                    "audit checkpoint features were not finite");
        }
        // Feature contract spot checks: warmth is zero on a fresh root, the checkpoint multiple
        // is echoed, and the legal-move count matches the initial position.
        require(auditResult.stop_features[0][16] == 0.0 &&
                    auditResult.stop_features[1][15] == 1.0 &&
                    auditResult.stop_features[0][11] == 20.0,
                "audit checkpoint features broke the binding contract");

        // A certain predictor behind a permissive guard stops at the first checkpoint.
        const SelfPlaySearchParameters openParameters(
            16, openPolicy(certainPredictorPath, {0.5, 0.5, 0.5}), treeSearchParameters(), 0.3F,
            0.0F);
        ChessSelfPlaySearch openSearch(runtimeParameters, openParameters, inferenceParameters);
        const auto stoppedResult =
            openSearch.search({productionRequest(openSearch, {8, 16, 24})}).results.front();
        require(stoppedResult.stop_reason == SearchStopReason::LearnedEarlyStop &&
                    stoppedResult.stop_checkpoint_index == 0 && stoppedResult.final_visits == 8,
                "a certain predictor did not stop at the first checkpoint");
        require(stoppedResult.checkpoints.size() == 1 && stoppedResult.stop_verdicts[0] == 1,
                "an early-stopped search recorded checkpoints past its stop");

        // The same open policy in shadow mode records the verdict but completes the cap.
        const auto shadowResult =
            openSearch.search({productionRequest(openSearch, {8, 16, 24}, true)}).results.front();
        require(shadowResult.final_visits == 32 && shadowResult.stop_checkpoint_index == -1 &&
                    shadowResult.stop_verdicts[0] == 1,
                "shadow mode did not record the verdict while completing the cap");

        // An uncertain predictor never stops and the search reaches the cap.
        const SelfPlaySearchParameters cautiousParameters(
            16, openPolicy(uncertainPredictorPath, {0.5, 0.5, 0.5}), treeSearchParameters(), 0.3F,
            0.0F);
        ChessSelfPlaySearch cautiousSearch(runtimeParameters, cautiousParameters,
                                           inferenceParameters);
        const auto cautiousResult =
            cautiousSearch.search({productionRequest(cautiousSearch, {8, 16, 24})}).results.front();
        require(cautiousResult.stop_reason == SearchStopReason::CapReached &&
                    cautiousResult.final_visits == 32,
                "an uncertain predictor stopped a search");

        // A tiny movement guard blocks stopping even for a certain predictor.
        const SelfPlaySearchParameters guardedParameters(
            16, openPolicy(certainPredictorPath, {0.5, 0.5, 0.5}, 1e-12), treeSearchParameters(),
            0.3F, 0.0F);
        ChessSelfPlaySearch guardedSearch(runtimeParameters, guardedParameters,
                                          inferenceParameters);
        const auto guardedResult =
            guardedSearch.search({productionRequest(guardedSearch, {8, 16, 24})}).results.front();
        require(guardedResult.stop_reason == SearchStopReason::CapReached &&
                    std::ranges::all_of(guardedResult.stop_probabilities,
                                        [](const double value) { return value == -1.0; }),
                "the movement guard did not block the predictor");

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

        // Retained root: an audit on a warm root captures the zeroth checkpoint at starting
        // visits, so warmth and the movement basis reflect the retained tree.
        ChessSelfPlaySearchRequest retained{
            .root = checkpointResult.root,
            .assigned_additional_visits = std::nullopt,
            .policy_checkpoint_visits = {88, 96, 104},
            .parallel_searches = std::nullopt,
            .add_root_noise = false,
            .force_root_playouts = false,
            .checkpoint_detail = SearchCheckpointDetail::Policies,
            .root_ply = 12,
            .audit = true,
        };
        const auto warmResult = search.search({retained}).results.front();
        require(warmResult.starting_visits == 80 && warmResult.final_visits == 112,
                "warm audit did not search the cap in additional visits");
        require(warmResult.stop_features[0][16] == 5.0 && warmResult.stop_features[0][12] == 12.0,
                "warm audit features did not carry warmth and ply");

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
