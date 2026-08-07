#pragma once

#include "search/InferenceModel.hpp"
#include "search/InferenceTypes.hpp"
#include "util/Timing.hpp"
#include "util/py.hpp"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

// Executes asynchronous model batches and converts their tensors into game-legal predictions.

struct InferenceOutput {
    torch::Tensor policies;
    torch::Tensor outcomes;
};

template <typename Game>
[[nodiscard]] SearchInferenceResult<Game>
processInferencePosition(const float *policy, const float *outcome,
                         const typename Game::Position &position) {
    const float win = outcome[static_cast<std::size_t>(WdlIndex::Win)];
    const float draw = outcome[static_cast<std::size_t>(WdlIndex::Draw)];
    const float loss = outcome[static_cast<std::size_t>(WdlIndex::Loss)];
    if (!std::isfinite(win) || !std::isfinite(draw) || !std::isfinite(loss) || win < 0.0F ||
        draw < 0.0F || loss < 0.0F || std::abs(win + draw + loss - 1.0F) > 1e-2F) {
        throw std::runtime_error("Inference model WDL output must be three probabilities");
    }

    const std::vector<typename Game::Action> legalActions = Game::legalActions(position);
    std::vector<std::pair<typename Game::Action, float>> actions;
    actions.reserve(legalActions.size());
    float legalPolicySum = 0.0F;
    for (const typename Game::Action action : legalActions) {
        const int actionId = Game::actionId(action, position);
        if (actionId < 0 ||
            static_cast<std::size_t>(actionId) >= Game::inferenceDimensions().actions) {
            throw std::logic_error("Game contract produced an action outside its policy space");
        }
        const float probability = policy[actionId];
        if (!std::isfinite(probability) || probability < 0.0F) {
            throw std::runtime_error("Inference model policy must contain finite probabilities");
        }
        actions.emplace_back(action, probability);
        legalPolicySum += probability;
    }
    std::ranges::sort(actions, {}, [&position](const auto &actionProbability) {
        return Game::actionId(actionProbability.first, position);
    });
    if (!actions.empty() && legalPolicySum < 1e-5F) {
        const float uniformProbability = 1.0F / static_cast<float>(actions.size());
        for (auto &[action, probability] : actions) {
            static_cast<void>(action);
            probability = uniformProbability;
        }
    } else if (legalPolicySum >= 1e-5F) {
        for (auto &[action, probability] : actions) {
            static_cast<void>(action);
            probability /= legalPolicySum;
        }
        std::erase_if(actions, [](const auto &actionProbability) {
            return !(actionProbability.second > 0.0F);
        });
    }
    return {
        .actions = std::move(actions),
        .outcome = {.win = win, .draw = draw, .loss = loss},
    };
}

class InferenceRunner {
public:
    InferenceRunner(const std::string &modelPath, InferenceDevice device, int deviceId,
                    size_t maximumBatchSize, bool useDedicatedCudaStream,
                    InferenceDimensions dimensions);

    [[nodiscard]] torch::Tensor createInputBuffer() const;
    [[nodiscard]] InferenceOutput createOutputBuffer() const;
    void forwardInto(const torch::Tensor &encodedBoards, size_t batchSize, InferenceOutput &output);
    [[nodiscard]] PreparedInferenceModel prepareModelRefresh(const std::string &modelPath) const;
    void commitModelRefresh(PreparedInferenceModel updatedModel) noexcept;

    [[nodiscard]] size_t maximumBatchSize() const noexcept { return m_maximumBatchSize; }
    [[nodiscard]] bool usesCuda() const noexcept { return m_device.is_cuda(); }

private:
    const torch::Device m_device;
    const torch::Dtype m_torchDtype;
    const size_t m_maximumBatchSize;
    const InferenceDimensions m_dimensions;
    PreparedInferenceModel m_model;
    torch::Tensor m_deviceInput;
    std::vector<torch::jit::IValue> m_modelInputs{1};
#ifdef USE_CUDA
    std::optional<at::cuda::CUDAStream> m_cudaStream;
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
                      InferenceDimensions dimensions);
    ~InferencePipeline();

    InferencePipeline(const InferencePipeline &) = delete;
    InferencePipeline &operator=(const InferencePipeline &) = delete;

    [[nodiscard]] WritableBatch acquireWritableBatch();
    void discardWritableBatch(size_t slotIndex);
    void submit(size_t slotIndex, size_t batchSize);
    [[nodiscard]] bool isCompleted(size_t slotIndex) const;
    template <typename Game, typename PositionAt>
    [[nodiscard]] std::vector<SearchInferenceResult<Game>>
    consume(size_t slotIndex, size_t positionCount, PositionAt positionAt) {
        const InferenceOutput output = waitCompletedOutput(slotIndex);
        try {
            Stopwatch processingTimer;
            std::vector<SearchInferenceResult<Game>> results;
            results.reserve(positionCount);
            const float *policies = output.policies.data_ptr<float>();
            const float *outcomes = output.outcomes.data_ptr<float>();
            const InferenceDimensions dimensions = Game::inferenceDimensions();
            for (const auto row : range(positionCount)) {
                results.push_back(processInferencePosition<Game>(
                    policies + row * dimensions.actions, outcomes + row * dimensions.outcomes,
                    positionAt(row)));
            }
            m_resultProcessingNanoseconds += processingTimer.elapsedNanoseconds();
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
        return m_inferenceNanoseconds.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t resultProcessingNanoseconds() const noexcept {
        return m_resultProcessingNanoseconds;
    }
    [[nodiscard]] std::uint64_t consumerWaitNanoseconds() const noexcept {
        return m_consumerWaitNanoseconds;
    }

private:
    enum class SlotState : uint8_t { Empty, Filling, Ready, Running, Complete, Failed, Stopped };

    struct Slot {
        torch::Tensor input;
        InferenceOutput output;
        std::exception_ptr exception;
        size_t batchSize = 0;
        std::atomic<SlotState> state = SlotState::Empty;
    };

    InferenceRunner m_runner;
    std::vector<std::unique_ptr<Slot>> m_slots;
    size_t m_producerCursor = 0;
    size_t m_consumerCursor = 0;
    std::atomic<bool> m_stopping = false;
    std::atomic<std::uint64_t> m_inferenceNanoseconds = 0;
    std::uint64_t m_resultProcessingNanoseconds = 0;
    std::uint64_t m_consumerWaitNanoseconds = 0;
    std::thread m_inferenceThread;

    void inferenceLoop();
    [[nodiscard]] InferenceOutput waitCompletedOutput(size_t slotIndex);
    void release(size_t slotIndex);
    [[nodiscard]] Slot &slotAt(size_t slotIndex);
    [[nodiscard]] const Slot &slotAt(size_t slotIndex) const;
};
