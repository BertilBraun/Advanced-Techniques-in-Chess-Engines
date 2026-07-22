#include "DirectInference.hpp"

#include "InferenceModel.hpp"

namespace {
torch::Device resolveDevice(const InferenceDevice requestedDevice, const int deviceId) {
    const bool useCuda = requestedDevice == InferenceDevice::Cuda ||
                         (requestedDevice == InferenceDevice::Auto && torch::cuda::is_available());
    if (!useCuda) {
        return torch::Device(torch::kCPU);
    }
    if (!torch::cuda::is_available()) {
        throw std::invalid_argument("CUDA inference requested but CUDA is unavailable");
    }
    if (deviceId < 0 || deviceId >= torch::cuda::device_count()) {
        throw std::invalid_argument("Invalid CUDA device ID");
    }
    return torch::Device(torch::kCUDA, deviceId);
}

torch::Dtype dtypeForDevice(const torch::Device &device) {
    return device.is_cuda() ? torch::kBFloat16 : torch::kFloat32;
}
} // namespace

DirectInferenceRunner::DirectInferenceRunner(const std::string &modelPath,
                                             const InferenceDevice device, const int deviceId,
                                             const size_t maximumBatchSize,
                                             const bool useDedicatedCudaStream)
    : m_device(resolveDevice(device, deviceId)), m_torchDtype(dtypeForDevice(m_device)),
      m_maximumBatchSize(maximumBatchSize),
      m_model(loadInferenceModel(modelPath, m_device, m_torchDtype)) {
    if (maximumBatchSize == 0) {
        throw std::invalid_argument("Maximum direct inference batch size must be positive");
    }
#ifdef USE_CUDA
    if (m_device.is_cuda() && useDedicatedCudaStream) {
        m_cudaStream = at::cuda::getStreamFromPool(false, m_device.index());
    }
#else
    if (m_device.is_cuda() && useDedicatedCudaStream) {
        throw std::runtime_error("Dedicated CUDA streams require a CUDA-enabled native build");
    }
#endif
    m_deviceInput =
        torch::empty({static_cast<int64_t>(maximumBatchSize), BOARD_C, BOARD_LEN, BOARD_LEN},
                     torch::TensorOptions().device(m_device).dtype(m_torchDtype));
}

torch::Tensor DirectInferenceRunner::createInputBuffer() const {
    return torch::empty(
        {static_cast<int64_t>(m_maximumBatchSize), BOARD_C, BOARD_LEN, BOARD_LEN},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt8).pinned_memory(usesCuda()));
}

DirectInferenceOutput DirectInferenceRunner::createOutputBuffer() const {
    const torch::TensorOptions options =
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32).pinned_memory(usesCuda());
    return {torch::empty({static_cast<int64_t>(m_maximumBatchSize), ACTION_SIZE}, options),
            torch::empty({static_cast<int64_t>(m_maximumBatchSize), 3}, options)};
}

void DirectInferenceRunner::forwardInto(const torch::Tensor &encodedBoards, const size_t batchSize,
                                        DirectInferenceOutput &output) {
    if (batchSize == 0 || batchSize > m_maximumBatchSize) {
        throw std::invalid_argument("Direct inference batch size is outside runner capacity");
    }
    if (encodedBoards.device().is_cuda() || encodedBoards.scalar_type() != torch::kInt8 ||
        encodedBoards.dim() != 4 || encodedBoards.size(0) < static_cast<int64_t>(batchSize) ||
        encodedBoards.size(1) != BOARD_C || encodedBoards.size(2) != BOARD_LEN ||
        encodedBoards.size(3) != BOARD_LEN) {
        throw std::invalid_argument("Direct inference input must be a CPU int8 board batch");
    }
    if (output.policies.device().is_cuda() || output.policies.scalar_type() != torch::kFloat32 ||
        output.policies.dim() != 2 || output.policies.size(0) < static_cast<int64_t>(batchSize) ||
        output.policies.size(1) != ACTION_SIZE || output.outcomes.device().is_cuda() ||
        output.outcomes.scalar_type() != torch::kFloat32 || output.outcomes.dim() != 2 ||
        output.outcomes.size(0) < static_cast<int64_t>(batchSize) || output.outcomes.size(1) != 3) {
        throw std::invalid_argument("Direct inference output buffers have invalid shapes or types");
    }

    torch::InferenceMode inferenceMode;
#ifdef USE_CUDA
    std::optional<at::cuda::CUDAStreamGuard> streamGuard;
    if (m_cudaStream.has_value()) {
        streamGuard.emplace(*m_cudaStream);
    }
#endif
    const torch::Tensor source = encodedBoards.narrow(0, 0, static_cast<int64_t>(batchSize));
    torch::Tensor deviceInput = m_deviceInput.narrow(0, 0, static_cast<int64_t>(batchSize));
    deviceInput.copy_(source, usesCuda());

    m_modelInputs[0] = deviceInput;
    const torch::jit::IValue modelOutput = m_model.forward(m_modelInputs);
    const auto outputTuple = modelOutput.toTuple();
    if (outputTuple->elements().size() != 2) {
        throw std::runtime_error("Inference model must return policy and WDL tensors");
    }
    const torch::Tensor policies = outputTuple->elements()[0].toTensor();
    const torch::Tensor outcomes = outputTuple->elements()[1].toTensor();
    output.policies.narrow(0, 0, static_cast<int64_t>(batchSize)).copy_(policies, usesCuda());
    output.outcomes.narrow(0, 0, static_cast<int64_t>(batchSize)).copy_(outcomes, usesCuda());
#ifdef USE_CUDA
    if (usesCuda()) {
        at::cuda::getCurrentCUDAStream(m_device.index()).synchronize();
    }
#endif
}

DirectInferencePipeline::DirectInferencePipeline(const std::string &modelPath,
                                                 const InferenceDevice device, const int deviceId,
                                                 const size_t maximumBatchSize,
                                                 const size_t slotCount,
                                                 const bool useDedicatedCudaStream)
    : m_runner(modelPath, device, deviceId, maximumBatchSize, useDedicatedCudaStream) {
    if (slotCount < 2) {
        throw std::invalid_argument("Direct inference pipeline requires at least two slots");
    }
    m_slots.reserve(slotCount);
    for (size_t index = 0; index < slotCount; ++index) {
        auto slot = std::make_unique<Slot>();
        slot->input = m_runner.createInputBuffer();
        slot->output = m_runner.createOutputBuffer();
        m_slots.push_back(std::move(slot));
    }
    m_inferenceThread = std::thread(&DirectInferencePipeline::inferenceLoop, this);
}

DirectInferencePipeline::~DirectInferencePipeline() {
    m_stopping.store(true, std::memory_order_release);
    for (const std::unique_ptr<Slot> &slot : m_slots) {
        slot->state.store(SlotState::Stopped, std::memory_order_release);
        slot->state.notify_all();
    }
    if (m_inferenceThread.joinable()) {
        m_inferenceThread.join();
    }
}

DirectInferencePipeline::WritableBatch DirectInferencePipeline::acquireWritableBatch() {
    Slot &slot = slotAt(m_producerCursor);
    SlotState state = slot.state.load(std::memory_order_acquire);
    while (state != SlotState::Empty) {
        if (state == SlotState::Stopped) {
            throw std::runtime_error("Direct inference pipeline is stopped");
        }
        slot.state.wait(state, std::memory_order_acquire);
        state = slot.state.load(std::memory_order_acquire);
    }
    slot.state.store(SlotState::Filling, std::memory_order_release);
    return {m_producerCursor, slot.input.data_ptr<int8>(), m_runner.maximumBatchSize()};
}

void DirectInferencePipeline::discardWritableBatch(const size_t slotIndex) {
    if (slotIndex != m_producerCursor) {
        throw std::invalid_argument("Direct inference batches must be discarded in order");
    }
    Slot &slot = slotAt(slotIndex);
    if (slot.state.load(std::memory_order_acquire) != SlotState::Filling) {
        throw std::logic_error("Direct inference slot was not acquired for writing");
    }
    slot.state.store(SlotState::Empty, std::memory_order_release);
    slot.state.notify_one();
}

void DirectInferencePipeline::submit(const size_t slotIndex, const size_t batchSize) {
    if (slotIndex != m_producerCursor) {
        throw std::invalid_argument("Direct inference batches must be submitted in order");
    }
    Slot &slot = slotAt(slotIndex);
    if (slot.state.load(std::memory_order_acquire) != SlotState::Filling) {
        throw std::logic_error("Direct inference slot was not acquired for writing");
    }
    if (batchSize == 0 || batchSize > m_runner.maximumBatchSize()) {
        throw std::invalid_argument("Direct inference batch size is outside pipeline capacity");
    }
    slot.batchSize = batchSize;
    slot.state.store(SlotState::Ready, std::memory_order_release);
    slot.state.notify_one();
    m_producerCursor = (m_producerCursor + 1) % m_slots.size();
}

DirectInferenceOutput DirectInferencePipeline::waitCompleted(const size_t slotIndex) {
    if (slotIndex != m_consumerCursor) {
        throw std::invalid_argument("Direct inference completions must be consumed in order");
    }
    Slot &slot = slotAt(slotIndex);
    SlotState state = slot.state.load(std::memory_order_acquire);
    while (state != SlotState::Complete && state != SlotState::Failed) {
        if (state == SlotState::Stopped) {
            throw std::runtime_error("Direct inference pipeline is stopped");
        }
        slot.state.wait(state, std::memory_order_acquire);
        state = slot.state.load(std::memory_order_acquire);
    }
    if (state == SlotState::Failed) {
        const std::exception_ptr exception = slot.exception;
        slot.exception = nullptr;
        slot.state.store(SlotState::Empty, std::memory_order_release);
        slot.state.notify_one();
        m_consumerCursor = (m_consumerCursor + 1) % m_slots.size();
        std::rethrow_exception(exception);
    }
    return {slot.output.policies.narrow(0, 0, static_cast<int64_t>(slot.batchSize)),
            slot.output.outcomes.narrow(0, 0, static_cast<int64_t>(slot.batchSize))};
}

void DirectInferencePipeline::release(const size_t slotIndex) {
    if (slotIndex != m_consumerCursor) {
        throw std::invalid_argument("Direct inference completions must be released in order");
    }
    Slot &slot = slotAt(slotIndex);
    if (slot.state.load(std::memory_order_acquire) != SlotState::Complete) {
        throw std::logic_error("Direct inference slot has not completed");
    }
    slot.state.store(SlotState::Empty, std::memory_order_release);
    slot.state.notify_one();
    m_consumerCursor = (m_consumerCursor + 1) % m_slots.size();
}

void DirectInferencePipeline::inferenceLoop() {
    size_t slotIndex = 0;
    while (!m_stopping.load(std::memory_order_acquire)) {
        Slot &slot = slotAt(slotIndex);
        SlotState state = slot.state.load(std::memory_order_acquire);
        while (state != SlotState::Ready) {
            if (state == SlotState::Stopped || m_stopping.load(std::memory_order_acquire)) {
                return;
            }
            slot.state.wait(state, std::memory_order_acquire);
            state = slot.state.load(std::memory_order_acquire);
        }
        slot.state.store(SlotState::Running, std::memory_order_release);
        try {
            const auto startedAt = std::chrono::steady_clock::now();
            m_runner.forwardInto(slot.input, slot.batchSize, slot.output);
            m_inferenceNanoseconds.fetch_add(
                static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                               std::chrono::steady_clock::now() - startedAt)
                                               .count()),
                std::memory_order_relaxed);
            slot.state.store(SlotState::Complete, std::memory_order_release);
        } catch (...) {
            slot.exception = std::current_exception();
            slot.state.store(SlotState::Failed, std::memory_order_release);
        }
        slot.state.notify_one();
        slotIndex = (slotIndex + 1) % m_slots.size();
    }
}

DirectInferencePipeline::Slot &DirectInferencePipeline::slotAt(const size_t slotIndex) {
    if (slotIndex >= m_slots.size()) {
        throw std::invalid_argument("Direct inference slot index is out of range");
    }
    return *m_slots[slotIndex];
}
