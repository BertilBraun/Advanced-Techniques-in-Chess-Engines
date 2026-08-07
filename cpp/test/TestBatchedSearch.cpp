#include "games/chess/ChessSelfPlaySearch.hpp"
#include "position.h"

namespace {
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
    const std::vector<const Board *> boards = {&board};
    return search.evaluate(boards).front().value();
}

} // namespace

int main() {
    Bitboards::init();
    Position::init();
    const std::filesystem::path modelPath =
        createTestModel("initial", 1.0F / 3.0F, 1.0F / 3.0F, 1.0F / 3.0F);
    const std::filesystem::path updatedModelPath = createTestModel("updated", 0.8F, 0.15F, 0.05F);
    const std::filesystem::path invalidModelPath =
        createTestModel("invalid", 0.5F, 0.5F, 0.0F, false);
    try {
        const InferenceRuntimeParameters runtimeParameters(0, modelPath.string(),
                                                           InferenceDevice::Cpu);
        const ChessSelfPlaySearchParameters searchParameters(1, 16, 8, 1.5F, 0.3F,
                                                             0.25F, 0);
        const BatchedInferenceParameters inferenceParameters(2, 4, 1);
        ChessSelfPlaySearch search(runtimeParameters, searchParameters, inferenceParameters, 7);
        const std::vector<std::uintptr_t> workerIdentities = search.workerIdentityTokens();
        require(workerIdentities.size() == 2, "direct search created the wrong worker count");
        require(search.modelVersion() == 7, "direct search lost its initial model version");
        require(std::abs(inferenceValue(search)) < 0.001F,
                "direct initial model returned the wrong value");
        const std::uint64_t evaluationsBeforeSearch =
            search.inferenceStatistics().first.evaluations;

        const std::vector<ChessSelfPlaySearchRequest> boards = {
            ChessSelfPlaySearchRequest(search.newRoot(Board{}), true),
            ChessSelfPlaySearchRequest(search.newRoot(Board{}), false),
        };
        const ChessSelfPlaySearchBatch results = search.search(boards, true);

        require(results.simulations_completed == 24,
                "direct scheduler completed the wrong search count");
        require(results.results.size() == 2, "direct scheduler returned the wrong root count");
        require(results.results[0].root.visits() == 16, "full root missed its exact visit limit");
        require(results.results[1].root.visits() == 8, "fast root missed its exact visit limit");
        require(results.results[0].root.virtualLoss() == 0.0F, "full root retained virtual loss");
        require(results.results[1].root.virtualLoss() == 0.0F, "fast root retained virtual loss");
        require(!results.results[0].visits.empty(), "full root did not expand legal moves");
        require(!results.results[1].visits.empty(), "fast root did not expand legal moves");

        const auto [statistics, timing] = search.inferenceStatistics();
        static_cast<void>(timing);
        require(statistics.evaluations == evaluationsBeforeSearch + 26,
                "direct scheduler did not distinguish root initialization from searches");
        require(statistics.modelInferencePositions == statistics.evaluations,
                "direct model-position accounting diverged from evaluations");
        require(statistics.modelInferenceCalls > 0, "direct scheduler recorded no model calls");

        const std::vector<ChessSelfPlaySearchRequest> completedBoards = {
            ChessSelfPlaySearchRequest(results.results[0].root, true),
            ChessSelfPlaySearchRequest(results.results[1].root, false),
        };
        const ChessSelfPlaySearchBatch completedResults = search.search(completedBoards, false);
        require(completedResults.simulations_completed == 0,
                "completed roots unexpectedly performed additional searches");

        const InferenceStatistics beforeRefresh = search.inferenceStatistics().first;
        search.refreshModel(8, updatedModelPath.string());
        require(search.modelVersion() == 8, "direct refresh did not publish its model version");
        require(search.workerIdentityTokens() == workerIdentities,
                "direct refresh reconstructed inference workers");
        require(std::abs(inferenceValue(search) - 0.75F) < 0.001F,
                "direct refresh retained old model output");
        require(search.inferenceStatistics().first.evaluations > beforeRefresh.evaluations,
                "direct refresh reset cumulative statistics");
        const ChessSelfPlaySearchBatch retainedResults = search.search(completedBoards, false);
        require(retainedResults.simulations_completed == 0,
                "pure model refresh discarded retained search roots");

        try {
            search.refreshModel(9, invalidModelPath.string());
            throw std::runtime_error("invalid direct model refresh unexpectedly succeeded");
        } catch (const std::invalid_argument &) {
        }
        require(search.modelVersion() == 8,
                "failed direct refresh published an unvalidated model version");
        require(search.workerIdentityTokens() == workerIdentities,
                "failed direct refresh reconstructed inference workers");
        require(std::abs(inferenceValue(search) - 0.75F) < 0.001F,
                "failed direct refresh changed active weights");

        std::vector<Board> concurrentBoards(200);
        std::vector<const Board *> concurrentBoardPointers;
        concurrentBoardPointers.reserve(concurrentBoards.size());
        for (const Board &board : concurrentBoards) {
            concurrentBoardPointers.push_back(&board);
        }
        std::atomic<bool> concurrentBatchStarted = false;
        std::future<std::vector<ChessInferenceResult>> concurrentInference =
            std::async(std::launch::async, [&] {
                concurrentBatchStarted.store(true, std::memory_order_release);
                return search.evaluate(concurrentBoardPointers);
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

        for (uint64 version = 10; version < 29; ++version) {
            const std::filesystem::path &refreshPath =
                version % 2 == 0 ? updatedModelPath : modelPath;
            search.refreshModel(version, refreshPath.string());
            require(search.workerIdentityTokens() == workerIdentities,
                    "repeated direct refresh changed worker identity");
        }
        require(search.modelVersion() == 28, "repeated direct refresh lost model version");

        const ChessSelfPlaySearchParameters sameCapacitySchedule(1, 16, 7, 1.25F, 0.3F,
                                                                 0.25F, 0);
        require(!search.updateSearchSchedule(sameCapacitySchedule),
                "equal-capacity schedule incorrectly required root replacement");
        const ChessSelfPlaySearchParameters largerSchedule(1, 24, 8, 1.25F, 0.3F,
                                                            0.25F, 0);
        require(search.updateSearchSchedule(largerSchedule),
                "larger schedule did not report an arena-capacity change");
        require(search.workerIdentityTokens() == workerIdentities,
                "schedule update reconstructed direct inference workers");

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
