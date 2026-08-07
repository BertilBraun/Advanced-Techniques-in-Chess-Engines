#include "games/chess/ChessGameContract.hpp"
#include "search/InferencePipeline.hpp"

namespace {
std::filesystem::path createTestModel() {
    torch::jit::script::Module model("inference_pipeline_test");
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
        ("inference-pipeline-test-" + std::to_string(uniqueSuffix) + ".jit.pt");
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
    const std::filesystem::path modelPath = createTestModel();
    try {
        InferenceRunner runner(modelPath.string(), InferenceDevice::Cpu, 0, 4, false,
                               ChessGameContract::inferenceDimensions());
        torch::Tensor input = runner.createInputBuffer();
        input.zero_();
        InferenceOutput output = runner.createOutputBuffer();
        runner.forwardInto(input, 3, output);
        require(output.policies.size(0) == 4, "runner changed reusable policy capacity");
        require(std::abs(output.policies[0].sum().item<float>() - 1.0F) < 0.001F,
                "runner returned invalid policy");

        InferencePipeline pipeline(modelPath.string(), InferenceDevice::Cpu, 0, 4, 2, false,
                                   ChessGameContract::inferenceDimensions());
        const InferencePipeline::WritableBatch first = pipeline.acquireWritableBatch();
        std::memset(first.data, 0,
                    first.capacity * ChessRepresentationDimensions::channel_count *
                        ChessRepresentationDimensions::board_length *
                        ChessRepresentationDimensions::board_length);
        pipeline.submit(first.slotIndex, 2);
        const auto readinessDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!pipeline.isCompleted(first.slotIndex) &&
               std::chrono::steady_clock::now() < readinessDeadline) {
            std::this_thread::yield();
        }
        require(pipeline.isCompleted(first.slotIndex),
                "pipeline did not publish nonblocking completion readiness");
        const InferenceOutput completed = pipeline.waitCompleted(first.slotIndex);
        require(completed.policies.size(0) == 2, "pipeline returned wrong policy batch size");
        require(completed.outcomes.size(0) == 2, "pipeline returned wrong outcome batch size");
        pipeline.release(first.slotIndex);

        const InferencePipeline::WritableBatch second = pipeline.acquireWritableBatch();
        require(second.slotIndex != first.slotIndex, "pipeline did not advance through its slots");
        std::memset(second.data, 0,
                    second.capacity * ChessRepresentationDimensions::channel_count *
                        ChessRepresentationDimensions::board_length *
                        ChessRepresentationDimensions::board_length);
        pipeline.submit(second.slotIndex, 4);
        require(!pipeline.isCompleted(first.slotIndex),
                "pipeline reported a released slot as completed");
        static_cast<void>(pipeline.waitCompleted(second.slotIndex));
        pipeline.release(second.slotIndex);
    } catch (...) {
        std::filesystem::remove(modelPath);
        throw;
    }
    std::filesystem::remove(modelPath);
    return 0;
}
