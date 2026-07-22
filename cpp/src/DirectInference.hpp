#pragma once

#include "BoardEncoding.hpp"
#include "InferenceClientTypes.hpp"

#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

struct DirectInferenceOutput {
    torch::Tensor policies;
    torch::Tensor outcomes;
};

class DirectInferenceRunner {
public:
    DirectInferenceRunner(const std::string &modelPath, InferenceDevice device, int deviceId,
                          size_t maximumBatchSize, bool useDedicatedCudaStream);

    [[nodiscard]] torch::Tensor createInputBuffer() const;
    [[nodiscard]] DirectInferenceOutput createOutputBuffer() const;
    void forwardInto(const torch::Tensor &encodedBoards, size_t batchSize,
                     DirectInferenceOutput &output);

    [[nodiscard]] size_t maximumBatchSize() const noexcept { return m_maximumBatchSize; }
    [[nodiscard]] bool usesCuda() const noexcept { return m_device.is_cuda(); }

private:
    const torch::Device m_device;
    const torch::Dtype m_torchDtype;
    const size_t m_maximumBatchSize;
    torch::jit::script::Module m_model;
    torch::Tensor m_deviceInput;
    std::vector<torch::jit::IValue> m_modelInputs{1};
#ifdef USE_CUDA
    std::optional<at::cuda::CUDAStream> m_cudaStream;
#endif
};

class DirectInferencePipeline {
public:
    struct WritableBatch {
        size_t slotIndex;
        int8 *data;
        size_t capacity;
    };

    DirectInferencePipeline(const std::string &modelPath, InferenceDevice device, int deviceId,
                            size_t maximumBatchSize, size_t slotCount, bool useDedicatedCudaStream);
    ~DirectInferencePipeline();

    DirectInferencePipeline(const DirectInferencePipeline &) = delete;
    DirectInferencePipeline &operator=(const DirectInferencePipeline &) = delete;

    [[nodiscard]] WritableBatch acquireWritableBatch();
    void discardWritableBatch(size_t slotIndex);
    void submit(size_t slotIndex, size_t batchSize);
    [[nodiscard]] bool isCompleted(size_t slotIndex) const;
    [[nodiscard]] DirectInferenceOutput waitCompleted(size_t slotIndex);
    void release(size_t slotIndex);
    [[nodiscard]] std::uint64_t inferenceNanoseconds() const noexcept {
        return m_inferenceNanoseconds.load(std::memory_order_relaxed);
    }

private:
    enum class SlotState : uint8_t { Empty, Filling, Ready, Running, Complete, Failed, Stopped };

    struct Slot {
        torch::Tensor input;
        DirectInferenceOutput output;
        std::exception_ptr exception;
        size_t batchSize = 0;
        std::atomic<SlotState> state = SlotState::Empty;
    };

    DirectInferenceRunner m_runner;
    std::vector<std::unique_ptr<Slot>> m_slots;
    size_t m_producerCursor = 0;
    size_t m_consumerCursor = 0;
    std::atomic<bool> m_stopping = false;
    std::atomic<std::uint64_t> m_inferenceNanoseconds = 0;
    std::thread m_inferenceThread;

    void inferenceLoop();
    [[nodiscard]] Slot &slotAt(size_t slotIndex);
    [[nodiscard]] const Slot &slotAt(size_t slotIndex) const;
};
