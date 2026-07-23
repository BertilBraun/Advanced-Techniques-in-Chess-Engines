#include "MCTS/MCTS.hpp"
#include "position.h"

namespace {
std::filesystem::path createTestModel(const std::string &name, const float win, const float draw,
                                      const float loss, const bool validOutput = true) {
    torch::jit::script::Module model("direct_self_play_test");
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
        ("direct-self-play-test-" + name + "-" + std::to_string(uniqueSuffix) + ".jit.pt");
    model.save(path.string());
    return path;
}

void require(const bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

float inferenceValue(MCTS &search) {
    Board board;
    const std::vector<const Board *> boards = {&board};
    return search.inferenceBatch(boards).front().value();
}

void requireRefreshSemantics(const bool useInferenceCache,
                             const std::filesystem::path &initialModel,
                             const std::filesystem::path &updatedModel,
                             const std::filesystem::path &invalidModel) {
    const InferenceClientParams clientParameters(0, initialModel.string(), 4, 0,
                                                 useInferenceCache ? 16 : 0, InferenceDevice::Cpu);
    const MCTSParams searchParameters(1, 16, 8, 1.5F, 0.3F, 0.25F, 0, 1);
    MCTS search(clientParameters, searchParameters, useInferenceCache, std::nullopt, 4);
    require(std::abs(inferenceValue(search)) < 0.001F,
            "queued client initial model returned the wrong value");
    search.refreshModel(5, updatedModel.string());
    require(search.modelVersion() == 5, "queued client did not publish refreshed model version");
    require(std::abs(inferenceValue(search) - 0.75F) < 0.001F,
            "queued client retained old model output after refresh");
    const InferenceStatistics beforeFailure = search.getInferenceStatistics().first;
    try {
        search.refreshModel(6, invalidModel.string());
        throw std::runtime_error("invalid queued model refresh unexpectedly succeeded");
    } catch (const std::invalid_argument &) {
    }
    require(search.modelVersion() == 5,
            "failed queued refresh published an unvalidated model version");
    require(std::abs(inferenceValue(search) - 0.75F) < 0.001F,
            "failed queued refresh changed the active model");
    require(search.getInferenceStatistics().first.evaluations > beforeFailure.evaluations,
            "queued model refresh reset cumulative inference statistics");
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
        const InferenceClientParams clientParameters(0, modelPath.string(), 4, 0, 0,
                                                     InferenceDevice::Cpu);
        const MCTSParams searchParameters(1, 16, 8, 1.5F, 0.3F, 0.25F, 0, 1);
        const DirectSelfPlayInferenceParams directParameters(2, 4, 1);
        MCTS search(clientParameters, searchParameters, false, directParameters, 7);
        const std::vector<std::uintptr_t> workerIdentities = search.directWorkerIdentityTokens();
        require(workerIdentities.size() == 2, "direct search created the wrong worker count");
        require(search.modelVersion() == 7, "direct search lost its initial model version");
        require(std::abs(inferenceValue(search)) < 0.001F,
                "direct initial model returned the wrong value");
        const std::uint64_t evaluationsBeforeSearch =
            search.getInferenceStatistics().first.evaluations;

        const std::vector<MCTSBoard> boards = {
            MCTSBoard(search.newRoot(Board{}), true),
            MCTSBoard(search.newRoot(Board{}), false),
        };
        const MCTSResults results = search.search(boards, true);

        require(results.searchesCompleted == 24,
                "direct scheduler completed the wrong search count");
        require(results.results.size() == 2, "direct scheduler returned the wrong root count");
        require(results.results[0].root.visits() == 16, "full root missed its exact visit limit");
        require(results.results[1].root.visits() == 8, "fast root missed its exact visit limit");
        require(results.results[0].root.virtualLoss() == 0.0F, "full root retained virtual loss");
        require(results.results[1].root.virtualLoss() == 0.0F, "fast root retained virtual loss");
        require(!results.results[0].visits.empty(), "full root did not expand legal moves");
        require(!results.results[1].visits.empty(), "fast root did not expand legal moves");

        const auto [statistics, timing] = search.getInferenceStatistics();
        static_cast<void>(timing);
        require(statistics.evaluations == evaluationsBeforeSearch + 26,
                "direct scheduler did not distinguish root initialization from searches");
        require(statistics.modelInferencePositions == statistics.evaluations,
                "direct model-position accounting diverged from evaluations");
        require(statistics.modelInferenceCalls > 0, "direct scheduler recorded no model calls");

        const std::vector<MCTSBoard> completedBoards = {
            MCTSBoard(results.results[0].root, true),
            MCTSBoard(results.results[1].root, false),
        };
        const MCTSResults completedResults = search.search(completedBoards, false);
        require(completedResults.searchesCompleted == 0,
                "completed roots unexpectedly performed additional searches");

        const InferenceStatistics beforeRefresh = search.getInferenceStatistics().first;
        search.refreshModel(8, updatedModelPath.string());
        require(search.modelVersion() == 8, "direct refresh did not publish its model version");
        require(search.directWorkerIdentityTokens() == workerIdentities,
                "direct refresh reconstructed inference workers");
        require(std::abs(inferenceValue(search) - 0.75F) < 0.001F,
                "direct refresh retained old model output");
        require(search.getInferenceStatistics().first.evaluations > beforeRefresh.evaluations,
                "direct refresh reset cumulative statistics");
        const MCTSResults retainedResults = search.search(completedBoards, false);
        require(retainedResults.searchesCompleted == 0,
                "pure model refresh discarded retained search roots");

        try {
            search.refreshModel(9, invalidModelPath.string());
            throw std::runtime_error("invalid direct model refresh unexpectedly succeeded");
        } catch (const std::invalid_argument &) {
        }
        require(search.modelVersion() == 8,
                "failed direct refresh published an unvalidated model version");
        require(search.directWorkerIdentityTokens() == workerIdentities,
                "failed direct refresh reconstructed inference workers");
        require(std::abs(inferenceValue(search) - 0.75F) < 0.001F,
                "failed direct refresh changed active weights");

        std::atomic<bool> beginConcurrentInference = false;
        std::vector<std::future<void>> concurrentInference;
        for (int worker = 0; worker < 8; ++worker) {
            concurrentInference.push_back(std::async(std::launch::async, [&] {
                while (!beginConcurrentInference.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                for (int request = 0; request < 25; ++request) {
                    const float value = inferenceValue(search);
                    const bool usedInitialModel = std::abs(value) < 0.001F;
                    const bool usedUpdatedModel = std::abs(value - 0.75F) < 0.001F;
                    require(usedInitialModel || usedUpdatedModel,
                            "concurrent inference observed mixed model parameters and buffers");
                }
            }));
        }
        beginConcurrentInference.store(true, std::memory_order_release);
        for (uint64 version = 9; version < 29; ++version) {
            const std::filesystem::path &refreshPath =
                version % 2 == 0 ? updatedModelPath : modelPath;
            search.refreshModel(version, refreshPath.string());
            require(search.directWorkerIdentityTokens() == workerIdentities,
                    "repeated direct refresh changed worker identity");
        }
        for (std::future<void> &inference : concurrentInference) {
            inference.get();
        }
        require(search.modelVersion() == 28, "repeated direct refresh lost model version");

        const MCTSParams sameCapacitySchedule(1, 16, 7, 1.25F, 0.3F, 0.25F, 0, 1);
        require(!search.updateSearchSchedule(sameCapacitySchedule),
                "equal-capacity schedule incorrectly required root replacement");
        const MCTSParams largerSchedule(1, 24, 8, 1.25F, 0.3F, 0.25F, 0, 1);
        require(search.updateSearchSchedule(largerSchedule),
                "larger schedule did not report an arena-capacity change");
        require(search.directWorkerIdentityTokens() == workerIdentities,
                "schedule update reconstructed direct inference workers");

        requireRefreshSemantics(false, modelPath, updatedModelPath, invalidModelPath);
        requireRefreshSemantics(true, modelPath, updatedModelPath, invalidModelPath);
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
