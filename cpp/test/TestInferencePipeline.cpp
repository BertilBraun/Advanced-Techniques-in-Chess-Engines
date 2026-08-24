#include "TestRunner.hpp"
#include "games/chess/ChessGame.hpp"
#include "search/InferencePipeline.hpp"

namespace {
std::filesystem::path createTestModel() {
    torch::jit::script::Module model("inference_pipeline_test");
    model.define(R"JIT(
        def forward(self, boards):
            batch_size = boards.size(0)
            torch._assert(torch.all(boards >= 0), "negative test input")
            policies = torch.zeros((batch_size, )JIT" +
                 std::to_string(ChessEncoding::actionCount) + R"JIT(), device=boards.device)
            wins = torch.clamp(boards[:, 0, 0, 0].float(), 0.0, 1.0)
            draws = torch.zeros_like(wins)
            outcomes = torch.stack((wins, draws, 1.0 - wins), 1)
            search_correction = torch.full((batch_size, 1), 0.5, device=boards.device)
            return policies, outcomes, search_correction
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

void fillBatch(const InferencePipeline::WritableBatch &batch, const std::int8_t value) {
    std::memset(batch.data, value,
                batch.capacity * ChessRepresentationDimensions::channelCount *
                    ChessRepresentationDimensions::boardLength *
                    ChessRepresentationDimensions::boardLength);
}

void testTwoOutstandingBatches(const std::filesystem::path &modelPath, const InferenceDevice device,
                               const bool dedicatedCudaStream) {
    InferencePipeline pipeline(modelPath.string(), device, 0, 4, 2, dedicatedCudaStream,
                               ChessGame::Encoding::inferenceDimensions());
    const InferencePipeline::WritableBatch first = pipeline.acquireWritableBatch();
    fillBatch(first, 0);
    pipeline.submit(first.slotIndex, 2);
    const InferencePipeline::WritableBatch second = pipeline.acquireWritableBatch();
    fillBatch(second, 1);
    pipeline.submit(second.slotIndex, 2);

    require(!pipeline.isCompleted(second.slotIndex),
            "pipeline exposed a later completion before FIFO consumption");
    const auto readinessDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!pipeline.isCompleted(first.slotIndex) &&
           std::chrono::steady_clock::now() < readinessDeadline) {
        std::this_thread::yield();
    }
    require(pipeline.isCompleted(first.slotIndex),
            "pipeline did not publish nonblocking completion readiness");
    const std::vector<Board> positions(2);
    const std::vector<SearchInferenceResult<ChessGame>> firstCompleted =
        pipeline.consume<ChessGame>(first.slotIndex, positions);
    require(firstCompleted[0].outcome.loss == 1.0F,
            "first outstanding batch used the wrong input buffer");
    const std::vector<SearchInferenceResult<ChessGame>> secondCompleted =
        pipeline.consume<ChessGame>(second.slotIndex, positions);
    require(secondCompleted[0].outcome.win == 1.0F,
            "second outstanding batch used the wrong input buffer");

    const InferencePipeline::WritableBatch reused = pipeline.acquireWritableBatch();
    require(reused.slotIndex == first.slotIndex, "pipeline did not reuse the released first slot");
    pipeline.discardWritableBatch(reused.slotIndex);
}

void testFailureReleasesSlot(const std::filesystem::path &modelPath) {
    InferencePipeline pipeline(modelPath.string(), InferenceDevice::Cpu, 0, 2, 2, false,
                               ChessGame::Encoding::inferenceDimensions());
    const InferencePipeline::WritableBatch failing = pipeline.acquireWritableBatch();
    fillBatch(failing, -1);
    pipeline.submit(failing.slotIndex, 1);
    bool propagatedFailure = false;
    try {
        pipeline.consumeWithoutResult(failing.slotIndex);
    } catch (const std::exception &exception) {
        propagatedFailure =
            std::string(exception.what()).find("negative test input") != std::string::npos;
    }
    require(propagatedFailure, "pipeline did not propagate inference failure");

    const InferencePipeline::WritableBatch next = pipeline.acquireWritableBatch();
    fillBatch(next, 0);
    pipeline.submit(next.slotIndex, 1);
    pipeline.consumeWithoutResult(next.slotIndex);
    const InferencePipeline::WritableBatch reused = pipeline.acquireWritableBatch();
    require(reused.slotIndex == failing.slotIndex,
            "pipeline did not release a failed slot for reuse");
    pipeline.discardWritableBatch(reused.slotIndex);
}
} // namespace

int runInferencePipelineTests() {
    const std::filesystem::path modelPath = createTestModel();
    try {
        InferenceRunner runner(modelPath.string(), InferenceDevice::Cpu, 0, 4, false,
                               ChessGame::Encoding::inferenceDimensions());
        torch::Tensor input = runner.createInputBuffer();
        input.zero_();
        InferenceOutput output = runner.createOutputBuffer();
        runner.forwardInto(input, 3, output);
        require(output.policies.size(0) == 4, "runner changed reusable policy capacity");
        require(torch::isfinite(output.policies[0]).all().item<bool>(),
                "runner returned nonfinite policy logits");

        bool rejectedCudaBackendOnCpu = false;
        try {
            InferenceRunner explicitCudaRunner(modelPath.string(), InferenceDevice::Cpu, 0, 4,
                                               false, ChessGame::Encoding::inferenceDimensions(),
                                               SdpaBackend::MemoryEfficient);
        } catch (const std::invalid_argument &) {
            rejectedCudaBackendOnCpu = true;
        }
        require(rejectedCudaBackendOnCpu, "runner accepted an explicit CUDA SDPA backend on CPU");

        testTwoOutstandingBatches(modelPath, InferenceDevice::Cpu, false);
        testFailureReleasesSlot(modelPath);
#ifdef USE_CUDA
        if (torch::cuda::is_available()) {
            testTwoOutstandingBatches(modelPath, InferenceDevice::Cuda, true);
        }
#endif
    } catch (...) {
        std::filesystem::remove(modelPath);
        throw;
    }
    std::filesystem::remove(modelPath);
    return 0;
}
