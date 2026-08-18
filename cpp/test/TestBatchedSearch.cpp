#include "TestRunner.hpp"
#include "games/chess/presentation/ChessSearchPresentation.hpp"
#include "position.h"
#include "search/SelfPlay.hpp"

namespace {
using ChessSelfPlaySearch = GameSelfPlaySearch<ChessGame>;
using ChessSelfPlaySearchParameters = SelfPlaySearchParameters;
using ChessSelfPlaySearchRequest = SelfPlaySearchRequest<ChessGame>;
using ChessSelfPlaySearchBatch = SelfPlaySearchBatch<ChessGame>;
using ChessInferenceResult = SearchInferenceResult<ChessGame>;

TreeSearchParameters
treeSearchParameters(const float explorationConstant,
                     const FirstPlayUrgencyParameters firstPlayUrgency =
                         FirstPlayUrgencyParameters(FirstPlayUrgencyKind::Zero),
                     const float forcedPlayoutCoefficient = 0.0F,
                     const float valueDiscountPerPly = 1.0F) {
    return TreeSearchParameters(explorationConstant, firstPlayUrgency, forcedPlayoutCoefficient,
                                valueDiscountPerPly);
}

std::filesystem::path createTestModel(const std::string &name, const float win, const float draw,
                                      const float loss, const bool validOutput = true) {
    torch::jit::script::Module model("batched_search_test");
    model.register_parameter("outcome_parameter",
                             validOutput ? torch::tensor({win, draw}) : torch::tensor({win}),
                             false);
    model.register_buffer("outcome_buffer", torch::tensor({loss}));
    model.define(R"JIT(
        def forward(self, boards):
            batch_size = boards.size(0)
            policies = torch.ones((batch_size, 1880), device=boards.device) / 1880.0
            outcome = torch.cat((self.outcome_parameter, self.outcome_buffer))
            outcomes = outcome.unsqueeze(0).repeat((batch_size, 1))
            return policies, outcomes
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

float inferenceValue(ChessSelfPlaySearch &search) {
    Board board;
    const std::vector<Board> boards = {board};
    return search.evaluate(boards).front().value();
}

std::uint32_t totalVisits(const std::vector<GameSearchVisit> &visits) {
    return std::accumulate(visits.begin(), visits.end(), std::uint32_t{0},
                           [](const std::uint32_t total, const GameSearchVisit visit) {
                               return total + visit.visit_count;
                           });
}

ChessSelfPlaySearchBatch searchRequests(ChessSelfPlaySearch &search,
                                        const std::vector<bool> &fullSearches) {
    std::vector<ChessSelfPlaySearchRequest> requests;
    requests.reserve(fullSearches.size());
    for (const bool fullSearch : fullSearches) {
        requests.emplace_back(search.newRoot(Board{}), fullSearch);
    }
    ChessSelfPlaySearchBatch batch = search.search(requests);
    require(batch.results.size() == requests.size(),
            "staged search returned the wrong result count");
    for (const std::size_t index : range(requests.size())) {
        require(&batch.results[index].root.tree() == &requests[index].root.tree(),
                "staged search changed deterministic result ordering");
    }
    return batch;
}

} // namespace

int runBatchedSearchTests() {
    Stockfish::Bitboards::init();
    Stockfish::Position::init();
    const std::filesystem::path modelPath =
        createTestModel("initial", 1.0F / 3.0F, 1.0F / 3.0F, 1.0F / 3.0F);
    const std::filesystem::path updatedModelPath = createTestModel("updated", 0.8F, 0.15F, 0.05F);
    const std::filesystem::path invalidModelPath =
        createTestModel("invalid", 0.5F, 0.5F, 0.0F, false);
    try {
        try {
            static_cast<void>(
                FirstPlayUrgencyParameters(FirstPlayUrgencyKind::ReducedParentValue, -0.1F));
            throw std::runtime_error("negative FPU reduction unexpectedly validated");
        } catch (const std::invalid_argument &) {
        }
        for (const float invalidDiscount : {0.0F, 1.01F, std::numeric_limits<float>::infinity()}) {
            try {
                static_cast<void>(treeSearchParameters(
                    1.5F, FirstPlayUrgencyParameters(FirstPlayUrgencyKind::Zero), 0.0F,
                    invalidDiscount));
                throw std::runtime_error("invalid value discount unexpectedly validated");
            } catch (const std::invalid_argument &) {
            }
        }
        try {
            static_cast<void>(FirstPlayUrgencyParameters(FirstPlayUrgencyKind::Zero, 0.1F));
            throw std::runtime_error("zero FPU unexpectedly accepted a reduction");
        } catch (const std::invalid_argument &) {
        }
        try {
            static_cast<void>(
                FirstPlayUrgencyParameters(FirstPlayUrgencyKind::ReducedParentValue, 0.0F));
            throw std::runtime_error("reduced-parent FPU unexpectedly accepted zero reduction");
        } catch (const std::invalid_argument &) {
        }
        try {
            static_cast<void>(
                treeSearchParameters(1.5F, FirstPlayUrgencyParameters(FirstPlayUrgencyKind::Zero),
                                     std::numeric_limits<float>::infinity()));
            throw std::runtime_error("nonfinite forced-playout coefficient unexpectedly validated");
        } catch (const std::invalid_argument &) {
        }
        const InferenceConfiguration runtimeParameters(0, modelPath.string(), InferenceDevice::Cpu);
        const ChessSelfPlaySearchParameters searchParameters(1, 16, 8, treeSearchParameters(1.5F),
                                                             0.3F, 0.25F);
        const BatchedInferenceParameters inferenceParameters(2, 4, 1);
        ChessSelfPlaySearch search(runtimeParameters, searchParameters, inferenceParameters, 7);
        require(search.modelGeneration() == 7, "search lost its initial model generation");
        require(std::abs(inferenceValue(search)) < 0.001F,
                "initial model returned the wrong value");
        const std::uint64_t evaluationsBeforeSearch = search.inferenceStatistics().evaluations;

        const std::vector<ChessSelfPlaySearchRequest> boards = {
            ChessSelfPlaySearchRequest(search.newRoot(Board{}), true),
            ChessSelfPlaySearchRequest(search.newRoot(Board{}), false),
        };
        const ChessSelfPlaySearchBatch results = search.search(boards, true);

        require(results.simulations_completed == 24, "scheduler completed the wrong search count");
        require(results.results.size() == 2, "scheduler returned the wrong root count");
        require(results.results[0].root.visits() == 16, "full root missed its exact visit limit");
        require(results.results[1].root.visits() == 8, "fast root missed its exact visit limit");
        require(results.results[0].root.tree().root().virtual_loss == 0.0F,
                "full root retained virtual loss");
        require(results.results[1].root.tree().root().virtual_loss == 0.0F,
                "fast root retained virtual loss");
        require(!results.results[0].search_visits.empty(), "full root did not expand legal moves");
        require(!results.results[1].search_visits.empty(), "fast root did not expand legal moves");
        for (const auto &result : results.results) {
            require(std::ranges::all_of(
                        result.search_visits,
                        [](const GameSearchVisit &visit) { return visit.visit_count > 0; }),
                    "search result exposed zero-visit children");
            const auto highestVisit = std::ranges::max_element(
                result.search_visits, {},
                [](const GameSearchVisit &visit) { return visit.visit_count; });
            require(highestVisit != result.search_visits.end(),
                    "search did not expose a highest-visited child");
            require(result.highest_visited_child_visit_count == highestVisit->visit_count,
                    "search exposed the wrong highest-child visit count");
            const auto &rootNode = result.root.tree().root();
            const auto edge = std::ranges::find_if(rootNode.children, [&](const auto &candidate) {
                return ChessGame::Encoding::actionId(candidate.action, rootNode.position) ==
                       result.highest_visited_child_action_id;
            });
            require(edge != rootNode.children.end(),
                    "highest-visited child action is absent from the root");
            require(std::abs(result.highest_visited_child_q +
                             result.root.tree().valueDiscountPerPly() * edge->value_sum /
                                 static_cast<float>(edge->visits)) < 1e-6F,
                    "highest-visited child Q was not oriented to the root player");
        }
        require(results.results[0].search_visits == results.results[0].policy_target_visits,
                "disabled full search changed the policy target");
        require(results.results[1].search_visits == results.results[1].policy_target_visits,
                "disabled fast search changed the policy target");

        const InferenceStatistics statistics = search.inferenceStatistics();
        require(statistics.evaluations == evaluationsBeforeSearch + 26,
                "scheduler did not distinguish root initialization from searches");
        require(statistics.modelInferencePositions == statistics.evaluations,
                "model-position accounting diverged from evaluations");
        require(statistics.modelInferenceCalls > 0, "scheduler recorded no model calls");
        require(statistics.cacheTotalPositions == statistics.modelInferencePositions,
                "cache tracker did not observe every model position");
        require(statistics.cacheRepeatedHashes > 0,
                "identical roots did not produce cache upper-bound repeats");
        require(statistics.cacheUniqueHashes + statistics.cacheRepeatedHashes ==
                    statistics.cacheTotalPositions,
                "cache telemetry totals are inconsistent");

        const std::vector<ChessSelfPlaySearchRequest> completedBoards = {
            ChessSelfPlaySearchRequest(results.results[0].root, true),
            ChessSelfPlaySearchRequest(results.results[1].root, false),
        };
        const ChessSelfPlaySearchBatch completedResults = search.search(completedBoards, false);
        require(completedResults.simulations_completed == 0,
                "completed roots unexpectedly performed additional searches");

        const InferenceStatistics beforeRefresh = search.inferenceStatistics();
        search.refreshModel(8, updatedModelPath.string());
        require(search.inferenceStatistics().cacheTotalPositions == 0,
                "search model refresh retained cache observations");
        require(search.modelGeneration() == 8, "refresh did not publish its model generation");
        require(std::abs(inferenceValue(search) - 0.75F) < 0.001F,
                "refresh retained old model output");
        require(search.inferenceStatistics().evaluations > beforeRefresh.evaluations,
                "refresh reset cumulative statistics");
        require(completedBoards[0].root.visits() == 0 && completedBoards[1].root.visits() == 0,
                "model refresh retained stale search trees");
        const ChessSelfPlaySearchBatch refreshedResults = search.search(completedBoards, false);
        require(refreshedResults.simulations_completed == 24,
                "refreshed roots did not rebuild under the new model");

        try {
            search.refreshModel(9, invalidModelPath.string());
            throw std::runtime_error("invalid model refresh unexpectedly succeeded");
        } catch (const std::invalid_argument &) {
        }
        require(search.modelGeneration() == 8,
                "failed refresh published an unvalidated model version");
        require(std::abs(inferenceValue(search) - 0.75F) < 0.001F,
                "failed refresh changed active weights");

        std::vector<Board> concurrentBoards(200);
        std::atomic<bool> concurrentBatchStarted = false;
        std::future<std::vector<ChessInferenceResult>> concurrentInference =
            std::async(std::launch::async, [&] {
                concurrentBatchStarted.store(true, std::memory_order_release);
                return search.evaluate(concurrentBoards);
            });
        while (!concurrentBatchStarted.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        search.refreshModel(9, modelPath.string());
        const std::vector<ChessInferenceResult> concurrentResults = concurrentInference.get();
        require(!concurrentResults.empty(), "concurrent inference returned no results");
        const float concurrentValue = concurrentResults.front().value();
        const bool usedInitialModel = std::abs(concurrentValue) < 0.001F;
        const bool usedUpdatedModel = std::abs(concurrentValue - 0.75F) < 0.001F;
        require(usedInitialModel || usedUpdatedModel,
                "concurrent inference observed mixed model parameters and buffers");
        for (const ChessInferenceResult &result : concurrentResults) {
            require(std::abs(result.value() - concurrentValue) < 0.001F,
                    "one accepted batch crossed model generations");
        }

        for (std::uint64_t version = 10; version < 29; ++version) {
            const std::filesystem::path &refreshPath =
                version % 2 == 0 ? updatedModelPath : modelPath;
            search.refreshModel(version, refreshPath.string());
        }
        require(search.modelGeneration() == 28, "repeated refresh lost model generation");

        const ChessSelfPlaySearchParameters sameCapacitySchedule(
            1, 16, 7, treeSearchParameters(1.25F), 0.3F, 0.25F);
        require(!search.updateSearchSchedule(sameCapacitySchedule),
                "equal-capacity schedule incorrectly required root replacement");
        const ChessSelfPlaySearchParameters discountedSchedule(
            1, 16, 7,
            treeSearchParameters(1.25F, FirstPlayUrgencyParameters(FirstPlayUrgencyKind::Zero),
                                 0.0F, 0.9985F),
            0.3F, 0.25F);
        require(search.updateSearchSchedule(discountedSchedule),
                "value-discount change did not require root replacement");
        try {
            static_cast<void>(
                search.search({ChessSelfPlaySearchRequest(results.results[0].root, true)}));
            throw std::runtime_error("stale root unexpectedly accepted a new value discount");
        } catch (const std::invalid_argument &) {
        }
        const ChessSelfPlaySearchParameters largerSchedule(1, 24, 8, treeSearchParameters(1.25F),
                                                           0.3F, 0.25F);
        require(search.updateSearchSchedule(largerSchedule),
                "larger schedule did not report an arena-capacity change");

        const ChessSelfPlaySearchParameters forcedSchedule(
            1, 64, 16,
            treeSearchParameters(
                1.5F, FirstPlayUrgencyParameters(FirstPlayUrgencyKind::ReducedParentValue, 0.2F),
                2.0F),
            0.3F, 0.25F);
        ChessSelfPlaySearch forcedSearch(runtimeParameters, forcedSchedule, inferenceParameters);
        const ChessSelfPlaySearchBatch forcedResults =
            forcedSearch.search({ChessSelfPlaySearchRequest(forcedSearch.newRoot(Board{}), true),
                                 ChessSelfPlaySearchRequest(forcedSearch.newRoot(Board{}), false)});
        require(totalVisits(forcedResults.results[0].search_visits) == 64 &&
                    totalVisits(forcedResults.results[1].search_visits) == 16,
                "forced playouts changed authoritative actual visit totals");
        require(totalVisits(forcedResults.results[0].policy_target_visits) <= 64,
                "forced-playout target exceeded actual visits");
        require(forcedResults.results[1].search_visits ==
                    forcedResults.results[1].policy_target_visits,
                "fast search applied forced playouts or pruning");

        require(initialFastSearchAdmissionCount(8, 2, 4, 1, 4) == 2,
                "4:1 mixed admission selected the wrong initial fast count");
        require(initialFastSearchAdmissionCount(5, 2, 4, 1, 4) == 2,
                "non-integral mixed admission did not round up");
        require(initialFastSearchAdmissionCount(7, 2, 4, 4, 4) == 7,
                "equal budgets did not admit every fast search");
        require(initialFastSearchAdmissionCount(0, 3, 4, 1, 4) == 0,
                "all-full admission unexpectedly created fast work");
        require(initialFastSearchAdmissionCount(7, 0, 4, 1, 4) == 7,
                "all-fast admission did not admit every search");
        require(initialFastSearchAdmissionCount(384, 128, 600, 150, 256) == 128,
                "capacity-fill admission did not fill the configured inference slots");
        require(initialFastSearchAdmissionCount(384, 256, 600, 150, 256) == 96,
                "capacity-fill admission incorrectly displaced ratio-based work");

        const ChessSelfPlaySearchParameters stagedParameters(1, 4, 1, treeSearchParameters(1.5F),
                                                             0.3F, 0.25F);
        ChessSelfPlaySearch stagedSearch(runtimeParameters, stagedParameters, inferenceParameters);
        const std::vector<bool> mixedKinds = {false, true, false, false, true, false, false};
        const ChessSelfPlaySearchBatch mixedResults = searchRequests(stagedSearch, mixedKinds);
        require(mixedResults.simulations_completed == 13,
                "4:1 staged search completed the wrong simulation count");
        for (const std::size_t index : range(mixedKinds.size())) {
            const std::uint32_t expectedVisits = mixedKinds[index] ? 4U : 1U;
            require(mixedResults.results[index].root.visits() == expectedVisits,
                    "4:1 staged search missed a request budget");
        }

        ChessSelfPlaySearch allFullSearch(runtimeParameters, stagedParameters, inferenceParameters);
        const ChessSelfPlaySearchBatch allFullResults =
            searchRequests(allFullSearch, {true, true, true});
        require(allFullResults.simulations_completed == 12,
                "all-full search completed the wrong simulation count");

        ChessSelfPlaySearch allFastSearch(runtimeParameters, stagedParameters, inferenceParameters);
        const ChessSelfPlaySearchBatch allFastResults =
            searchRequests(allFastSearch, {false, false, false, false, false});
        require(allFastResults.simulations_completed == 5,
                "all-fast search completed the wrong simulation count");

        const ChessSelfPlaySearchParameters equalParameters(1, 4, 4, treeSearchParameters(1.5F),
                                                            0.3F, 0.25F);
        ChessSelfPlaySearch equalSearch(runtimeParameters, equalParameters, inferenceParameters);
        const ChessSelfPlaySearchBatch equalResults =
            searchRequests(equalSearch, {false, true, false, true});
        require(equalResults.simulations_completed == 16,
                "equal-budget search completed the wrong simulation count");

        ChessSelfPlaySearch terminalSearch(runtimeParameters, stagedParameters,
                                           inferenceParameters);
        std::vector<ChessSelfPlaySearchRequest> terminalRequests = {
            ChessSelfPlaySearchRequest(terminalSearch.newRoot(Board{}), true),
            ChessSelfPlaySearchRequest(terminalSearch.newRoot(Board{}), false),
            ChessSelfPlaySearchRequest(terminalSearch.newRoot(Board{}), false),
            ChessSelfPlaySearchRequest(terminalSearch.newRoot(Board{}), false),
            ChessSelfPlaySearchRequest(terminalSearch.newRoot(Board{}), false),
            ChessSelfPlaySearchRequest(
                terminalSearch.newRoot(Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")), false),
        };
        try {
            static_cast<void>(terminalSearch.search(terminalRequests));
            throw std::runtime_error("terminal staged search unexpectedly returned a move");
        } catch (const std::logic_error &error) {
            require(std::string(error.what()) == "Completed search has no visited child",
                    "terminal staged search changed its error semantics");
        }
        require(terminalRequests.back().root.visits() == 1,
                "waiting terminal request was not incrementally admitted");

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
