#pragma once

#include "common.hpp"
#include "games/GameConcepts.hpp"
#include "search/InferenceTypes.hpp"
#include "util/Timing.hpp"
#include "util/py.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#endif

// Executes asynchronous model batches and converts their tensors into game-legal predictions.

struct InferenceOutput {
    torch::Tensor policies;
    torch::Tensor outcomes;
    torch::Tensor search_corrections;
};

class InferenceCompletion {
public:
    explicit InferenceCompletion(bool usesCuda) noexcept : m_usesCuda(usesCuda) {}
    ~InferenceCompletion() noexcept;

    InferenceCompletion(const InferenceCompletion &) = delete;
    InferenceCompletion &operator=(const InferenceCompletion &) = delete;

    void record();
    [[nodiscard]] bool ready() const;
    void wait() const;
    void finishFailedSubmission() noexcept;

private:
    void waitWithoutThrowing() const noexcept;

    const bool m_usesCuda;
#ifdef USE_CUDA
    // Blocking sync: the default event spins, and the thread that waits here is the tree owner,
    // so a spinning wait costs a core per self-play process for the whole of every GPU batch.
    c10::cuda::CUDAEvent m_cudaEvent{cudaEventDisableTiming | cudaEventBlockingSync};
#endif
};

template <SearchGame Game>
[[nodiscard]] SearchInferenceResult<Game>
processInferencePosition(const float *policy, const float *outcome,
                         const float searchCorrection,
                         const typename Game::State &position);

using PreparedInferenceModel = std::unique_ptr<torch::jit::script::Module>;

// Name and shape of every parameter and buffer of the loaded model. Freezing folds them into
// constants, so the refresh contract is checked against this signature instead.
struct ModelTensorSignature {
    std::string name;
    std::vector<std::int64_t> sizes;

    [[nodiscard]] bool operator==(const ModelTensorSignature &) const = default;
};

#ifdef USE_CUDA
struct CapturedInferenceGraph {
    std::size_t batch_size;
    std::unique_ptr<at::cuda::CUDAGraph> graph;
};
#endif

class InferenceRunner {
public:
    InferenceRunner(const std::string &modelPath, InferenceDevice device, int deviceId,
                    size_t maximumBatchSize, bool useDedicatedCudaStream,
                    InferenceDimensions dimensions, bool allowGraphCapture,
                    SdpaBackend sdpaBackend = SdpaBackend::Automatic);

    [[nodiscard]] torch::Tensor createInputBuffer() const;
    [[nodiscard]] InferenceOutput createOutputBuffer() const;
    void forwardInto(const torch::Tensor &encodedBoards, size_t batchSize, InferenceOutput &output);
    [[nodiscard]] PreparedInferenceModel prepareModelRefresh(const std::string &modelPath) const;
    void commitModelRefresh(PreparedInferenceModel updatedModel) noexcept;

    [[nodiscard]] size_t maximumBatchSize() const noexcept { return m_maximumBatchSize; }
    [[nodiscard]] bool usesCuda() const noexcept { return m_device.is_cuda(); }

private:
    friend class InferencePipeline;

    [[nodiscard]] torch::Tensor createDeviceInputBuffer() const;
    void stageOutput(const torch::Tensor &modelOutput, torch::Tensor &staging,
                     torch::Tensor &destination, size_t batchSize);
    void runEagerModel(size_t batchSize, InferenceOutput &output);
    void copyStagedOutput(size_t batchSize, InferenceOutput &output);
    void captureBatchGraphs();
    void releaseBatchGraphs() noexcept;
    [[nodiscard]] size_t capturedBatchSize(size_t batchSize) const noexcept;
    void forwardInto(const torch::Tensor &encodedBoards, size_t batchSize,
                     InferenceOutput &output, InferenceCompletion &completion);

    const torch::Device m_device;
    const torch::Dtype m_torchDtype;
    const size_t m_maximumBatchSize;
    const InferenceDimensions m_dimensions;
    const bool m_allowGraphCapture;
    PreparedInferenceModel m_model;
    torch::Tensor m_deviceInput;
    // Every dtype-changing copy across the host boundary allocates a device temporary and launches
    // a cast kernel; these persistent buffers make both copies same-dtype and allocation free.
    torch::Tensor m_deviceTypedInput;
    InferenceOutput m_deviceOutputStaging;
    std::vector<torch::jit::IValue> m_modelInputs{1};
    std::vector<ModelTensorSignature> m_parameterSignature;
    std::vector<ModelTensorSignature> m_bufferSignature;
#ifdef USE_CUDA
    std::optional<at::cuda::CUDAStream> m_cudaStream;
    // One replayable graph per batch bucket: a 12-layer tower costs milliseconds of host dispatch
    // per call and microseconds to replay, so the submitting thread stops starving the device.
    std::vector<CapturedInferenceGraph> m_batchGraphs;
#endif
};

class InferencePipeline {
public:
    struct WritableBatch {
        size_t slotIndex;
        std::int8_t *data;
        size_t capacity;
    };

    InferencePipeline(const std::string &modelPath, InferenceDevice device, int deviceId,
                      size_t maximumBatchSize, size_t slotCount, bool useDedicatedCudaStream,
                      InferenceDimensions dimensions, bool allowGraphCapture,
                      SdpaBackend sdpaBackend = SdpaBackend::Automatic);
    ~InferencePipeline();

    InferencePipeline(const InferencePipeline &) = delete;
    InferencePipeline &operator=(const InferencePipeline &) = delete;

    [[nodiscard]] WritableBatch acquireWritableBatch();
    void discardWritableBatch(size_t slotIndex);
    void submit(size_t slotIndex, size_t batchSize);
    [[nodiscard]] bool isCompleted(size_t slotIndex) const;
    template <SearchGame Game, typename Positions>
    [[nodiscard]] std::vector<SearchInferenceResult<Game>> consume(size_t slotIndex,
                                                                   const Positions &positions) {
        const InferenceOutput output = waitCompletedOutput(slotIndex);
        try {
            ScopedNanosecondTimer processingTimer(m_statistics.result_processing_nanoseconds);
            std::vector<SearchInferenceResult<Game>> results;
            results.reserve(positions.size());
            const float *policies = output.policies.data_ptr<float>();
            const float *outcomes = output.outcomes.data_ptr<float>();
            const float *searchCorrections = output.search_corrections.data_ptr<float>();
            const InferenceDimensions dimensions = Game::Encoding::inferenceDimensions();
            for (const auto row : range(positions.size())) {
                results.push_back(processInferencePosition<Game>(
                    policies + row * dimensions.actions, outcomes + row * dimensions.outcomes,
                    searchCorrections[row], positions[row]));
            }
            release(slotIndex);
            return results;
        } catch (...) {
            release(slotIndex);
            throw;
        }
    }
    void consumeWithoutResult(size_t slotIndex);
    [[nodiscard]] PreparedInferenceModel prepareModelRefresh(const std::string &modelPath) const;
    void commitModelRefresh(PreparedInferenceModel updatedModel) noexcept;
    [[nodiscard]] std::uint64_t inferenceNanoseconds() const noexcept {
        return m_statistics.inference_nanoseconds.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t resultProcessingNanoseconds() const noexcept {
        return m_statistics.result_processing_nanoseconds;
    }
    [[nodiscard]] std::uint64_t consumerWaitNanoseconds() const noexcept {
        return m_statistics.consumer_wait_nanoseconds;
    }

private:
    enum class SlotState : uint8_t { Empty, Filling, Ready, Running, Complete, Failed, Stopped };

    struct Slot {
        explicit Slot(const bool usesCuda) : completion(usesCuda) {}

        torch::Tensor input;
        InferenceOutput output;
        // Destruction drains the event before the preceding slot buffers are released.
        InferenceCompletion completion;
        std::exception_ptr exception;
        size_t batchSize = 0;
        std::atomic<SlotState> state = SlotState::Empty;
    };

    struct PipelineStatistics {
        std::atomic<std::uint64_t> inference_nanoseconds = 0;
        std::uint64_t result_processing_nanoseconds = 0;
        std::uint64_t consumer_wait_nanoseconds = 0;
    };

    InferenceRunner m_runner;
    std::vector<std::unique_ptr<Slot>> m_slots;
    size_t m_producerCursor = 0;
    size_t m_consumerCursor = 0;
    std::atomic<bool> m_stopping = false;
    PipelineStatistics m_statistics;
    std::thread m_inferenceThread;

    void inferenceLoop();
    [[nodiscard]] InferenceOutput waitCompletedOutput(size_t slotIndex);
    void release(size_t slotIndex);
    void resetSlot(size_t slotIndex);
    [[noreturn]] void releaseAndRethrow(size_t slotIndex, std::exception_ptr exception);
    [[nodiscard]] Slot &slotAt(size_t slotIndex);
    [[nodiscard]] const Slot &slotAt(size_t slotIndex) const;
};

template <SearchGame Game>
[[nodiscard]] SearchInferenceResult<Game>
processInferencePosition(const float *policy, const float *outcome,
                         const float searchCorrection,
                         const typename Game::State &position) {
    const float win = outcome[static_cast<std::size_t>(WdlIndex::Win)];
    const float draw = outcome[static_cast<std::size_t>(WdlIndex::Draw)];
    const float loss = outcome[static_cast<std::size_t>(WdlIndex::Loss)];
    if (!std::isfinite(win) || !std::isfinite(draw) || !std::isfinite(loss) || win < 0.0F ||
        draw < 0.0F || loss < 0.0F || std::abs(win + draw + loss - 1.0F) > 1e-2F) {
        throw std::runtime_error("Inference model WDL output must be three probabilities");
    }
    if (!std::isfinite(searchCorrection) || searchCorrection < 0.0F || searchCorrection > 1.0F) {
        throw std::runtime_error("Inference model search correction must lie in [0, 1]");
    }

    const std::vector<typename Game::Action> legalActions = Game::legalActions(position);
    std::vector<ScoredAction<typename Game::Action>> actions;
    actions.reserve(legalActions.size());
    float maximumLegalLogit = -std::numeric_limits<float>::infinity();
    for (const typename Game::Action action : legalActions) {
        const int actionId = Game::Encoding::actionId(action, position);
        if (actionId < 0 ||
            static_cast<std::size_t>(actionId) >= Game::Encoding::inferenceDimensions().actions) {
            throw std::logic_error("Game contract produced an action outside its policy space");
        }
        const float logit = policy[actionId];
        if (!std::isfinite(logit)) {
            throw std::runtime_error(
                "Inference model policy logits must be finite for legal actions");
        }
        actions.push_back({.action = action, .action_id = actionId, .prior = logit});
        maximumLegalLogit = std::max(maximumLegalLogit, logit);
    }
    std::ranges::sort(actions, {}, &ScoredAction<typename Game::Action>::action_id);
    if (!actions.empty()) {
        float exponentialSum = 0.0F;
        for (ScoredAction<typename Game::Action> &scored : actions) {
            scored.prior = std::exp(scored.prior - maximumLegalLogit);
            exponentialSum += scored.prior;
        }
        assert(std::isfinite(exponentialSum) && exponentialSum > 0.0F);
        for (ScoredAction<typename Game::Action> &scored : actions) {
            scored.prior /= exponentialSum;
        }
    }
    return {
        .actions = std::move(actions),
        .outcome = {.win = win, .draw = draw, .loss = loss},
        .search_correction = searchCorrection,
    };
}
