#include "DirectInference.hpp"

namespace {
std::filesystem::path createTestModel() {
    torch::jit::script::Module model("direct_inference_test");
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
        ("direct-inference-test-" + std::to_string(uniqueSuffix) + ".jit.pt");
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
        DirectInferenceRunner runner(modelPath.string(), InferenceDevice::Cpu, 0, 4, false);
        torch::Tensor input = runner.createInputBuffer();
        input.zero_();
        DirectInferenceOutput output = runner.createOutputBuffer();
        runner.forwardInto(input, 3, output);
        require(output.policies.size(0) == 4, "runner changed reusable policy capacity");
        require(std::abs(output.policies[0].sum().item<float>() - 1.0F) < 0.001F,
                "runner returned invalid policy");

        DirectInferencePipeline pipeline(modelPath.string(), InferenceDevice::Cpu, 0, 4, 2, false);
        const DirectInferencePipeline::WritableBatch first = pipeline.acquireWritableBatch();
        std::memset(first.data, 0, first.capacity * BOARD_C * BOARD_LEN * BOARD_LEN);
        pipeline.submit(first.slotIndex, 2);
        const DirectInferenceOutput completed = pipeline.waitCompleted(first.slotIndex);
        require(completed.policies.size(0) == 2, "pipeline returned wrong policy batch size");
        require(completed.outcomes.size(0) == 2, "pipeline returned wrong outcome batch size");
        pipeline.release(first.slotIndex);

        const DirectInferencePipeline::WritableBatch second = pipeline.acquireWritableBatch();
        require(second.slotIndex != first.slotIndex, "pipeline did not advance through its slots");
        std::memset(second.data, 0, second.capacity * BOARD_C * BOARD_LEN * BOARD_LEN);
        pipeline.submit(second.slotIndex, 4);
        static_cast<void>(pipeline.waitCompleted(second.slotIndex));
        pipeline.release(second.slotIndex);
    } catch (...) {
        std::filesystem::remove(modelPath);
        throw;
    }
    std::filesystem::remove(modelPath);
    return 0;
}
