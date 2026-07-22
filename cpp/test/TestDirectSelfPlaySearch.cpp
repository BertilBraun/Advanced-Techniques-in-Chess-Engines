#include "MCTS/MCTS.hpp"
#include "position.h"

namespace {
std::filesystem::path createTestModel() {
    torch::jit::script::Module model("direct_self_play_test");
    model.define(R"JIT(
        def forward(self, boards):
            batch_size = boards.size(0)
            policies = torch.ones((batch_size, 1880), device=boards.device) / 1880.0
            outcomes = torch.ones((batch_size, 3), device=boards.device) / 3.0
            return policies, outcomes
    )JIT");
    const auto uniqueSuffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("direct-self-play-test-" + std::to_string(uniqueSuffix) + ".jit.pt");
    model.save(path.string());
    return path;
}

void require(const bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
} // namespace

int main() {
    Bitboards::init();
    Position::init();
    const std::filesystem::path modelPath = createTestModel();
    try {
        const InferenceClientParams clientParameters(0, modelPath.string(), 4, 0, 0,
                                                     InferenceDevice::Cpu);
        const MCTSParams searchParameters(1, 16, 8, 1.5F, 0.3F, 0.25F, 0, 1);
        const DirectSelfPlayInferenceParams directParameters(1, 4, 1);
        MCTS search(clientParameters, searchParameters, false, directParameters);

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
        require(statistics.evaluations == 26,
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
    } catch (...) {
        std::filesystem::remove(modelPath);
        throw;
    }
    std::filesystem::remove(modelPath);
    return 0;
}
