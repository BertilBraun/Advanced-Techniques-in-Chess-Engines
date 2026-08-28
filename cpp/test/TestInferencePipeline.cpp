#include "TestRunner.hpp"
#include "games/chess/ChessGame.hpp"
#include "search/InferencePipeline.hpp"

namespace {
// The assertion reads a tensor back on the host, which is exactly what a stream capture forbids, so
// the graph tests need a model without it.
std::filesystem::path createTestModel(const bool assertNonNegativeInput = true) {
    torch::jit::script::Module model("inference_pipeline_test");
    model.define(R"JIT(
        def forward(self, boards):
            batch_size = boards.size(0))JIT" +
                 std::string(assertNonNegativeInput ? R"JIT(
            torch._assert(torch.all(boards >= 0), "negative test input"))JIT"
                                                    : "") +
                 R"JIT(
            policies = torch.zeros((batch_size, )JIT" +
                 std::to_string(ChessEncoding::actionCount) + R"JIT(), device=boards.device)
            wins = torch.clamp(boards[:, 0, 0, 0].float(), 0.0, 1.0)
            draws = torch.zeros_like(wins)
            outcomes = torch.stack((wins, draws, 1.0 - wins), 1)
            search_budget = torch.zeros((batch_size, 1), device=boards.device)
            return policies, outcomes, search_budget
    )JIT");
    const auto uniqueSuffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("inference-pipeline-test-" + std::to_string(uniqueSuffix) + ".jit.pt");
    model.save(path.string());
    return path;
}

// A refresh that keeps the captured graphs has to carry the new weights in through the frozen
// graph constants, so the refresh tests need a model whose output actually depends on a parameter.
std::filesystem::path createWeightedTestModel(const float winScale) {
    torch::jit::script::Module model("inference_pipeline_weighted_test");
    model.register_parameter("win_scale", torch::full({1}, winScale), false);
    model.define(R"JIT(
        def forward(self, boards):
            batch_size = boards.size(0)
            policies = torch.zeros((batch_size, )JIT" +
                 std::to_string(ChessEncoding::actionCount) + R"JIT(), device=boards.device)
            wins = torch.clamp(boards[:, 0, 0, 0].float() * self.win_scale, 0.0, 1.0)
            draws = torch.zeros_like(wins)
            outcomes = torch.stack((wins, draws, 1.0 - wins), 1)
            search_budget = torch.zeros((batch_size, 1), device=boards.device)
            return policies, outcomes, search_budget
    )JIT");
    const auto uniqueSuffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("inference-pipeline-weighted-test-" + std::to_string(uniqueSuffix) + ".jit.pt");
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

constexpr size_t REFRESH_COUNT = 4;

void testRepeatedModelRefresh(const std::filesystem::path &modelPath, const InferenceDevice device,
                              const bool dedicatedCudaStream) {
    InferenceRunner runner(modelPath.string(), device, 0, 4, dedicatedCudaStream,
                           ChessGame::Encoding::inferenceDimensions());
    torch::Tensor input = runner.createInputBuffer();
    InferenceOutput output = runner.createOutputBuffer();
#ifdef USE_CUDA
    const std::optional<at::cuda::MempoolId_t> capturePool = runner.graphPool();
    const size_t capturedGraphs = runner.capturedGraphCount();
    require(!runner.usesCuda() || capturedGraphs > 0,
            "runner captured no graphs for a capturable model");
#endif
    for (const auto refresh : range(REFRESH_COUNT)) {
        static_cast<void>(refresh);
        runner.commitModelRefresh(runner.prepareModelRefresh(modelPath.string()));
        input.fill_(1);
        runner.forwardInto(input, 3, output);
        require(output.outcomes[0][0].item<float>() == 1.0F,
                "runner did not serve the refreshed model");
#ifdef USE_CUDA
        // The refreshed graphs must land in the pool the first capture created; a second pool is
        // memory this process never gets back.
        require(runner.graphPool() == capturePool,
                "model refresh allocated an additional CUDA graph memory pool");
        require(runner.capturedGraphCount() == capturedGraphs,
                "model refresh changed the number of captured graphs");
#endif
    }
}

void testRefreshedWeightsAreServed(const std::filesystem::path &servingModelPath,
                                   const std::filesystem::path &silencedModelPath,
                                   const InferenceDevice device, const bool dedicatedCudaStream) {
    InferenceRunner runner(servingModelPath.string(), device, 0, 4, dedicatedCudaStream,
                           ChessGame::Encoding::inferenceDimensions());
    torch::Tensor input = runner.createInputBuffer();
    input.fill_(1);
    InferenceOutput output = runner.createOutputBuffer();
    runner.forwardInto(input, 3, output);
    require(output.outcomes[0][0].item<float>() == 1.0F,
            "runner did not serve the initial weights");

#ifdef USE_CUDA
    const size_t captures = runner.graphCaptureCount();
#endif
    runner.commitModelRefresh(runner.prepareModelRefresh(silencedModelPath.string()));
    runner.forwardInto(input, 3, output);
    require(output.outcomes[0][0].item<float>() == 0.0F,
            "model refresh did not change the weights the runner serves");

    runner.commitModelRefresh(runner.prepareModelRefresh(servingModelPath.string()));
    runner.forwardInto(input, 3, output);
    require(output.outcomes[0][0].item<float>() == 1.0F,
            "model refresh did not restore the original weights");
#ifdef USE_CUDA
    require(!runner.usesCuda() || runner.graphCaptureCount() == captures,
            "model refresh recaptured the inference graphs instead of replacing their weights");
#endif
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

#ifdef USE_CUDA
void testDisabledGraphsRefreshWithoutPool(const std::filesystem::path &modelPath) {
    setenv("ALPHAZERO_DISABLE_INFERENCE_GRAPHS", "1", 1);
    try {
        InferenceRunner runner(modelPath.string(), InferenceDevice::Cuda, 0, 4, true,
                               ChessGame::Encoding::inferenceDimensions());
        runner.commitModelRefresh(runner.prepareModelRefresh(modelPath.string()));
        require(runner.capturedGraphCount() == 0, "disabled inference graphs were captured anyway");
        require(!runner.graphPool().has_value(),
                "disabled inference graphs still claimed a memory pool");
        torch::Tensor input = runner.createInputBuffer();
        input.fill_(1);
        InferenceOutput output = runner.createOutputBuffer();
        runner.forwardInto(input, 3, output);
        require(output.outcomes[0][0].item<float>() == 1.0F,
                "eager fallback did not serve the refreshed model");
    } catch (...) {
        unsetenv("ALPHAZERO_DISABLE_INFERENCE_GRAPHS");
        throw;
    }
    unsetenv("ALPHAZERO_DISABLE_INFERENCE_GRAPHS");
}
#endif
} // namespace

int runInferencePipelineTests() {
    const std::filesystem::path modelPath = createTestModel();
    const std::filesystem::path capturableModelPath = createTestModel(false);
    const std::filesystem::path servingModelPath = createWeightedTestModel(1.0F);
    const std::filesystem::path silencedModelPath = createWeightedTestModel(0.0F);
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
            InferenceRunner explicitCudaRunner(
                modelPath.string(), InferenceDevice::Cpu, 0, 4, false,
                ChessGame::Encoding::inferenceDimensions(),
                InferenceExecutionOptions{.sdpa_backend = SdpaBackend::MemoryEfficient});
        } catch (const std::invalid_argument &) {
            rejectedCudaBackendOnCpu = true;
        }
        require(rejectedCudaBackendOnCpu, "runner accepted an explicit CUDA SDPA backend on CPU");

        bool rejectedChannelsLastOnCpu = false;
        try {
            InferenceRunner channelsLastRunner(
                modelPath.string(), InferenceDevice::Cpu, 0, 4, false,
                ChessGame::Encoding::inferenceDimensions(),
                InferenceExecutionOptions{.memory_format = InferenceMemoryFormat::ChannelsLast});
        } catch (const std::invalid_argument &) {
            rejectedChannelsLastOnCpu = true;
        }
        require(rejectedChannelsLastOnCpu, "runner accepted a channels-last memory format on CPU");

        // The default options must stay byte-for-byte the shipped path, or an unconfigured run
        // would silently change precision.
        require(InferenceExecutionOptions{} ==
                    InferenceExecutionOptions{.sdpa_backend = SdpaBackend::Automatic,
                                              .precision = InferencePrecision::BFloat16,
                                              .memory_format = InferenceMemoryFormat::Contiguous,
                                              .cudnn_benchmark = false},
                "default inference execution options changed");

        testTwoOutstandingBatches(modelPath, InferenceDevice::Cpu, false);
        testFailureReleasesSlot(modelPath);
        testRepeatedModelRefresh(capturableModelPath, InferenceDevice::Cpu, false);
        testRefreshedWeightsAreServed(servingModelPath, silencedModelPath, InferenceDevice::Cpu,
                                      false);
#ifdef USE_CUDA
        if (torch::cuda::is_available()) {
            testTwoOutstandingBatches(modelPath, InferenceDevice::Cuda, true);
            testRepeatedModelRefresh(capturableModelPath, InferenceDevice::Cuda, true);
            testRefreshedWeightsAreServed(servingModelPath, silencedModelPath,
                                          InferenceDevice::Cuda, true);
            testDisabledGraphsRefreshWithoutPool(capturableModelPath);
        }
#endif
    } catch (...) {
        std::filesystem::remove(modelPath);
        std::filesystem::remove(capturableModelPath);
        std::filesystem::remove(servingModelPath);
        std::filesystem::remove(silencedModelPath);
        throw;
    }
    std::filesystem::remove(modelPath);
    std::filesystem::remove(capturableModelPath);
    std::filesystem::remove(servingModelPath);
    std::filesystem::remove(silencedModelPath);
    return 0;
}
