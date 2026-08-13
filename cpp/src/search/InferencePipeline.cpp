#include "search/InferencePipeline.hpp"

#include "util/Timing.hpp"
#include "util/py.hpp"

namespace {
template <typename NamedTensorList>
void validateNamedTensors(const NamedTensorList &currentTensors,
                          const NamedTensorList &updatedTensors, const std::string &tensorKind) {
    if (currentTensors.size() != updatedTensors.size()) {
        throw std::invalid_argument("Updated inference model has a different number of " +
                                    tensorKind + "s");
    }

    auto current = currentTensors.begin();
    auto updated = updatedTensors.begin();
    while (current != currentTensors.end()) {
        const auto currentTensor = *current;
        const auto updatedTensor = *updated;
        if (currentTensor.name != updatedTensor.name) {
            throw std::invalid_argument("Updated inference model " + tensorKind +
                                        " names do not match");
        }
        if (currentTensor.value.sizes() != updatedTensor.value.sizes()) {
            throw std::invalid_argument("Updated inference model " + tensorKind + " " +
                                        currentTensor.name + " has a different shape");
        }
        ++current;
        ++updated;
    }
}

torch::jit::script::Module loadInferenceModel(const std::string &modelPath,
                                              const torch::Device &device,
                                              const torch::Dtype dataType) {
    std::string modelPathToLoad = modelPath;
    if (!modelPathToLoad.ends_with(".jit.pt") && !modelPathToLoad.ends_with(".pt")) {
        throw std::invalid_argument("Model path must end with '.jit.pt' or '.pt'");
    }
    if (!modelPathToLoad.ends_with(".jit.pt")) {
        modelPathToLoad = modelPathToLoad.substr(0, modelPathToLoad.size() - 3) + ".jit.pt";
    }
    if (!std::filesystem::exists(modelPathToLoad)) {
        throw std::invalid_argument("Model file does not exist: " + modelPathToLoad);
    }

    torch::jit::script::Module model = torch::jit::load(modelPathToLoad, device);
    model.to(dataType);
    model.eval();
    return model;
}

PreparedInferenceModel
prepareInferenceModelUpdate(const torch::jit::script::Module &model, const std::string &modelPath,
                            const torch::Device &device, const torch::Dtype dataType,
                            const torch::Tensor &validationInput, const std::int64_t actionCount,
                            const std::int64_t outcomeCount) {
    auto updatedModel = std::make_unique<torch::jit::script::Module>(
        loadInferenceModel(modelPath, device, dataType));
    const auto currentParameters = model.named_parameters();
    const auto updatedParameters = updatedModel->named_parameters();
    const auto currentBuffers = model.named_buffers();
    const auto updatedBuffers = updatedModel->named_buffers();
    validateNamedTensors(currentParameters, updatedParameters, "parameter");
    validateNamedTensors(currentBuffers, updatedBuffers, "buffer");
    torch::InferenceMode inferenceMode;
    const torch::jit::IValue output = updatedModel->forward({validationInput});
    if (!output.isTuple()) {
        throw std::invalid_argument("Updated inference model must return a tuple");
    }
    const auto outputTuple = output.toTuple();
    if (outputTuple->elements().size() != 2 || !outputTuple->elements()[0].isTensor() ||
        !outputTuple->elements()[1].isTensor()) {
        throw std::invalid_argument("Updated inference model must return policy and WDL tensors");
    }
    const torch::Tensor policy = outputTuple->elements()[0].toTensor();
    const torch::Tensor outcome = outputTuple->elements()[1].toTensor();
    if (policy.dim() != 2 || policy.size(0) != 1 || policy.size(1) != actionCount ||
        outcome.dim() != 2 || outcome.size(0) != 1 || outcome.size(1) != outcomeCount ||
        !torch::isfinite(policy).all().item<bool>() ||
        !torch::isfinite(outcome).all().item<bool>() || (policy < 0).any().item<bool>() ||
        (outcome < 0).any().item<bool>() || std::abs(policy.sum().item<float>() - 1.0F) > 1e-2F ||
        std::abs(outcome.sum().item<float>() - 1.0F) > 1e-2F) {
        throw std::invalid_argument("Updated inference model returned invalid output");
    }
    return updatedModel;
}

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

constexpr std::int64_t tensorSize(const std::size_t size) noexcept {
    return static_cast<std::int64_t>(size);
}
} // namespace

InferenceCompletion::~InferenceCompletion() noexcept { waitWithoutThrowing(); }

void InferenceCompletion::record() {
#ifdef USE_CUDA
    if (m_usesCuda) {
        m_cudaEvent.record();
    }
#endif
}

bool InferenceCompletion::ready() const {
#ifdef USE_CUDA
    if (m_usesCuda) {
        return m_cudaEvent.query();
    }
#endif
    return true;
}

void InferenceCompletion::wait() const {
#ifdef USE_CUDA
    if (m_usesCuda) {
        m_cudaEvent.synchronize();
    }
#endif
}

void InferenceCompletion::finishFailedSubmission() noexcept {
    // Drain work enqueued before the failure so its slot buffers can be reused safely.
    try {
        record();
        wait();
    } catch (...) {
    }
}

void InferenceCompletion::waitWithoutThrowing() const noexcept {
    try {
        wait();
    } catch (...) {
    }
}

InferenceRunner::InferenceRunner(const std::string &modelPath, const InferenceDevice device,
                                 const int deviceId, const size_t maximumBatchSize,
                                 const bool useDedicatedCudaStream,
                                 const InferenceDimensions dimensions)
    : m_device(resolveDevice(device, deviceId)), m_torchDtype(dtypeForDevice(m_device)),
      m_maximumBatchSize(maximumBatchSize), m_dimensions(dimensions),
      m_model(std::make_unique<torch::jit::script::Module>(
          loadInferenceModel(modelPath, m_device, m_torchDtype))) {
    if (maximumBatchSize == 0) {
        throw std::invalid_argument("Maximum inference batch size must be positive");
    }
    if (dimensions.channels == 0 || dimensions.rows == 0 || dimensions.columns == 0 ||
        dimensions.actions == 0 || dimensions.outcomes == 0) {
        throw std::invalid_argument("Inference dimensions must be positive");
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
    m_deviceInput = createDeviceInputBuffer();
}

torch::Tensor InferenceRunner::createInputBuffer() const {
    return torch::empty(
        {tensorSize(m_maximumBatchSize), tensorSize(m_dimensions.channels),
         tensorSize(m_dimensions.rows), tensorSize(m_dimensions.columns)},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt8).pinned_memory(usesCuda()));
}

torch::Tensor InferenceRunner::createDeviceInputBuffer() const {
    return torch::empty({tensorSize(m_maximumBatchSize), tensorSize(m_dimensions.channels),
                         tensorSize(m_dimensions.rows), tensorSize(m_dimensions.columns)},
                        torch::TensorOptions().device(m_device).dtype(m_torchDtype));
}

InferenceOutput InferenceRunner::createOutputBuffer() const {
    const torch::TensorOptions options =
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32).pinned_memory(usesCuda());
    return {
        .policies = torch::empty({tensorSize(m_maximumBatchSize), tensorSize(m_dimensions.actions)},
                                 options),
        .outcomes = torch::empty(
            {tensorSize(m_maximumBatchSize), tensorSize(m_dimensions.outcomes)}, options),
    };
}

void InferenceRunner::forwardInto(const torch::Tensor &encodedBoards,
                                  const torch::Tensor &deviceInputBuffer, const size_t batchSize,
                                  InferenceOutput &output, InferenceCompletion &completion) {
    if (batchSize == 0 || batchSize > m_maximumBatchSize) {
        throw std::invalid_argument("Inference batch size is outside runner capacity");
    }
    if (encodedBoards.device().is_cuda() || encodedBoards.scalar_type() != torch::kInt8 ||
        encodedBoards.dim() != 4 || encodedBoards.size(0) < static_cast<int64_t>(batchSize) ||
        encodedBoards.size(1) != tensorSize(m_dimensions.channels) ||
        encodedBoards.size(2) != tensorSize(m_dimensions.rows) ||
        encodedBoards.size(3) != tensorSize(m_dimensions.columns)) {
        throw std::invalid_argument("Inference input must be a CPU int8 board batch");
    }
    if (output.policies.device().is_cuda() || output.policies.scalar_type() != torch::kFloat32 ||
        output.policies.dim() != 2 || output.policies.size(0) < static_cast<int64_t>(batchSize) ||
        output.policies.size(1) != tensorSize(m_dimensions.actions) ||
        output.outcomes.device().is_cuda() || output.outcomes.scalar_type() != torch::kFloat32 ||
        output.outcomes.dim() != 2 || output.outcomes.size(0) < static_cast<int64_t>(batchSize) ||
        output.outcomes.size(1) != tensorSize(m_dimensions.outcomes)) {
        throw std::invalid_argument("Inference output buffers have invalid shapes or types");
    }

    torch::InferenceMode inferenceMode;
#ifdef USE_CUDA
    std::optional<at::cuda::CUDAStreamGuard> streamGuard;
    if (m_cudaStream.has_value()) {
        streamGuard.emplace(*m_cudaStream);
    }
#endif
    try {
        const torch::Tensor source = encodedBoards.narrow(0, 0, static_cast<int64_t>(batchSize));
        torch::Tensor deviceInput = deviceInputBuffer.narrow(0, 0, static_cast<int64_t>(batchSize));
        deviceInput.copy_(source, usesCuda());

        m_modelInputs[0] = deviceInput;
        const torch::jit::IValue modelOutput = m_model->forward(m_modelInputs);
        const auto outputTuple = modelOutput.toTuple();
        if (outputTuple->elements().size() != 2) {
            throw std::runtime_error("Inference model must return policy and WDL tensors");
        }
        const torch::Tensor policies = outputTuple->elements()[0].toTensor();
        const torch::Tensor outcomes = outputTuple->elements()[1].toTensor();
        output.policies.narrow(0, 0, static_cast<int64_t>(batchSize)).copy_(policies, usesCuda());
        output.outcomes.narrow(0, 0, static_cast<int64_t>(batchSize)).copy_(outcomes, usesCuda());
        completion.record();
    } catch (...) {
        completion.finishFailedSubmission();
        throw;
    }
}

void InferenceRunner::forwardInto(const torch::Tensor &encodedBoards, const size_t batchSize,
                                  InferenceOutput &output) {
    InferenceCompletion completion(usesCuda());
    forwardInto(encodedBoards, m_deviceInput, batchSize, output, completion);
    completion.wait();
}

PreparedInferenceModel InferenceRunner::prepareModelRefresh(const std::string &modelPath) const {
    const torch::Tensor validationInput =
        torch::zeros({1, tensorSize(m_dimensions.channels), tensorSize(m_dimensions.rows),
                      tensorSize(m_dimensions.columns)},
                     torch::TensorOptions().device(m_device).dtype(m_torchDtype));
    return prepareInferenceModelUpdate(*m_model, modelPath, m_device, m_torchDtype, validationInput,
                                       tensorSize(m_dimensions.actions),
                                       tensorSize(m_dimensions.outcomes));
}

void InferenceRunner::commitModelRefresh(PreparedInferenceModel updatedModel) noexcept {
    assert(m_model != nullptr);
    assert(updatedModel != nullptr);
    m_model.swap(updatedModel);
}

InferencePipeline::InferencePipeline(const std::string &modelPath, const InferenceDevice device,
                                     const int deviceId, const size_t maximumBatchSize,
                                     const size_t slotCount, const bool useDedicatedCudaStream,
                                     const InferenceDimensions dimensions)
    : m_runner(modelPath, device, deviceId, maximumBatchSize, useDedicatedCudaStream, dimensions) {
    if (slotCount < 2) {
        throw std::invalid_argument("Inference pipeline requires at least two slots");
    }
    m_slots.resize(slotCount);
    for (const auto index : range(slotCount)) {
        auto slot = std::make_unique<Slot>(m_runner.usesCuda());
        slot->input = m_runner.createInputBuffer();
        slot->deviceInput = m_runner.createDeviceInputBuffer();
        slot->output = m_runner.createOutputBuffer();
        m_slots[index] = std::move(slot);
    }
    m_inferenceThread = std::thread(&InferencePipeline::inferenceLoop, this);
}

InferencePipeline::~InferencePipeline() {
    m_stopping.store(true, std::memory_order_release);
    for (const std::unique_ptr<Slot> &slot : m_slots) {
        slot->state.store(SlotState::Stopped, std::memory_order_release);
        slot->state.notify_all();
    }
    if (m_inferenceThread.joinable()) {
        m_inferenceThread.join();
    }
}

InferencePipeline::WritableBatch InferencePipeline::acquireWritableBatch() {
    Slot &slot = slotAt(m_producerCursor);
    SlotState state = slot.state.load(std::memory_order_acquire);
    while (state != SlotState::Empty) {
        if (state == SlotState::Stopped) {
            throw std::runtime_error("Inference pipeline is stopped");
        }
        slot.state.wait(state, std::memory_order_acquire);
        state = slot.state.load(std::memory_order_acquire);
    }
    slot.state.store(SlotState::Filling, std::memory_order_release);
    return {
        .slotIndex = m_producerCursor,
        .data = slot.input.data_ptr<std::int8_t>(),
        .capacity = m_runner.maximumBatchSize(),
    };
}

void InferencePipeline::discardWritableBatch(const size_t slotIndex) {
    if (slotIndex != m_producerCursor) {
        throw std::invalid_argument("Inference batches must be discarded in order");
    }
    Slot &slot = slotAt(slotIndex);
    if (slot.state.load(std::memory_order_acquire) != SlotState::Filling) {
        throw std::logic_error("Inference slot was not acquired for writing");
    }
    slot.state.store(SlotState::Empty, std::memory_order_release);
    slot.state.notify_one();
}

void InferencePipeline::submit(const size_t slotIndex, const size_t batchSize) {
    if (slotIndex != m_producerCursor) {
        throw std::invalid_argument("Inference batches must be submitted in order");
    }
    Slot &slot = slotAt(slotIndex);
    if (slot.state.load(std::memory_order_acquire) != SlotState::Filling) {
        throw std::logic_error("Inference slot was not acquired for writing");
    }
    if (batchSize == 0 || batchSize > m_runner.maximumBatchSize()) {
        throw std::invalid_argument("Inference batch size is outside pipeline capacity");
    }
    slot.batchSize = batchSize;
    slot.state.store(SlotState::Ready, std::memory_order_release);
    slot.state.notify_one();
    m_producerCursor = (m_producerCursor + 1) % m_slots.size();
}

bool InferencePipeline::isCompleted(const size_t slotIndex) const {
    if (slotIndex != m_consumerCursor) {
        return false;
    }
    const SlotState state = slotAt(slotIndex).state.load(std::memory_order_acquire);
    return state == SlotState::Failed ||
           (state == SlotState::Complete && slotAt(slotIndex).completion.ready());
}

InferenceOutput InferencePipeline::waitCompletedOutput(const size_t slotIndex) {
    ScopedNanosecondTimer waitTimer(m_statistics.consumer_wait_nanoseconds);
    if (slotIndex != m_consumerCursor) {
        throw std::invalid_argument("Inference completions must be consumed in order");
    }
    Slot &slot = slotAt(slotIndex);
    SlotState state = slot.state.load(std::memory_order_acquire);
    while (state != SlotState::Complete && state != SlotState::Failed) {
        if (state == SlotState::Stopped) {
            throw std::runtime_error("Inference pipeline is stopped");
        }
        slot.state.wait(state, std::memory_order_acquire);
        state = slot.state.load(std::memory_order_acquire);
    }
    if (state == SlotState::Failed) {
        releaseAndRethrow(slotIndex, std::exchange(slot.exception, nullptr));
    }
    try {
        slot.completion.wait();
    } catch (...) {
        releaseAndRethrow(slotIndex, std::current_exception());
    }
    return {
        .policies = slot.output.policies.narrow(0, 0, static_cast<int64_t>(slot.batchSize)),
        .outcomes = slot.output.outcomes.narrow(0, 0, static_cast<int64_t>(slot.batchSize)),
    };
}

void InferencePipeline::consumeWithoutResult(const size_t slotIndex) {
    static_cast<void>(waitCompletedOutput(slotIndex));
    release(slotIndex);
}

void InferencePipeline::release(const size_t slotIndex) {
    if (slotIndex != m_consumerCursor) {
        throw std::invalid_argument("Inference completions must be released in order");
    }
    Slot &slot = slotAt(slotIndex);
    if (slot.state.load(std::memory_order_acquire) != SlotState::Complete) {
        throw std::logic_error("Inference slot has not completed");
    }
    resetSlot(slotIndex);
}

void InferencePipeline::resetSlot(const size_t slotIndex) {
    Slot &slot = slotAt(slotIndex);
    slot.state.store(SlotState::Empty, std::memory_order_release);
    slot.state.notify_one();
    m_consumerCursor = (m_consumerCursor + 1) % m_slots.size();
}

void InferencePipeline::releaseAndRethrow(const size_t slotIndex,
                                          const std::exception_ptr exception) {
    assert(exception != nullptr);
    resetSlot(slotIndex);
    std::rethrow_exception(exception);
}

PreparedInferenceModel InferencePipeline::prepareModelRefresh(const std::string &modelPath) const {
    for (const std::unique_ptr<Slot> &slot : m_slots) {
        if (slot->state.load(std::memory_order_acquire) != SlotState::Empty) {
            throw std::logic_error("Inference pipeline must be idle during model refresh");
        }
    }
    return m_runner.prepareModelRefresh(modelPath);
}

void InferencePipeline::commitModelRefresh(PreparedInferenceModel updatedModel) noexcept {
    m_runner.commitModelRefresh(std::move(updatedModel));
}

void InferencePipeline::inferenceLoop() {
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
            Stopwatch inferenceTimer;
            m_runner.forwardInto(slot.input, slot.deviceInput, slot.batchSize, slot.output,
                                 slot.completion);
            m_statistics.inference_nanoseconds.fetch_add(inferenceTimer.elapsedNanoseconds(),
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

InferencePipeline::Slot &InferencePipeline::slotAt(const size_t slotIndex) {
    if (slotIndex >= m_slots.size()) {
        throw std::invalid_argument("Inference slot index is out of range");
    }
    return *m_slots[slotIndex];
}

const InferencePipeline::Slot &InferencePipeline::slotAt(const size_t slotIndex) const {
    if (slotIndex >= m_slots.size()) {
        throw std::invalid_argument("Inference slot index is out of range");
    }
    return *m_slots[slotIndex];
}
