#include "search/InferencePipeline.hpp"

#include "util/Timing.hpp"
#include "util/py.hpp"

#include <ATen/Context.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>

namespace {
// Module::to(dtype) casts every parameter and buffer, integer ones included, unlike Python's
// nn.Module.to. A head that gathers through an index buffer would have its indices rounded to the
// inference dtype, and in bfloat16 an index of 4,094 is not even representable.
void adoptDataType(torch::jit::script::Module &model, const torch::Dtype dataType) {
    torch::NoGradGuard noGrad;
    for (const auto &parameter : model.named_parameters(true)) {
        if (parameter.value.is_floating_point()) {
            parameter.value.set_data(parameter.value.data().to(dataType));
        }
    }
    for (const auto &buffer : model.named_buffers(true)) {
        if (buffer.value.is_floating_point()) {
            buffer.value.set_data(buffer.value.data().to(dataType));
        }
    }
}

// Convolution weights must carry the activation layout: a channels-last activation against a
// contiguous weight makes cuDNN transpose the weight on every call instead of once here.
void adoptMemoryFormat(torch::jit::script::Module &model, const at::MemoryFormat memoryFormat) {
    if (memoryFormat == at::MemoryFormat::Contiguous) {
        return;
    }
    torch::NoGradGuard noGrad;
    for (const auto &parameter : model.named_parameters(true)) {
        if (parameter.value.dim() == 4) {
            parameter.value.set_data(parameter.value.data().contiguous(memoryFormat));
        }
    }
    for (const auto &buffer : model.named_buffers(true)) {
        if (buffer.value.dim() == 4) {
            buffer.value.set_data(buffer.value.data().contiguous(memoryFormat));
        }
    }
}

torch::jit::script::Module loadInferenceModel(const std::string &modelPath,
                                              const torch::Device &device,
                                              const torch::Dtype dataType,
                                              const at::MemoryFormat memoryFormat) {
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
    adoptDataType(model, dataType);
    adoptMemoryFormat(model, memoryFormat);
    model.eval();
    return model;
}

[[nodiscard]] std::vector<ModelTensorSignature>
tensorSignature(const torch::jit::named_parameter_list &tensors) {
    std::vector<ModelTensorSignature> signature;
    for (const auto &tensor : tensors) {
        signature.push_back({.name = tensor.name, .sizes = tensor.value.sizes().vec()});
    }
    return signature;
}

[[nodiscard]] std::vector<ModelTensorSignature>
tensorSignature(const torch::jit::named_buffer_list &tensors) {
    std::vector<ModelTensorSignature> signature;
    for (const auto &tensor : tensors) {
        signature.push_back({.name = tensor.name, .sizes = tensor.value.sizes().vec()});
    }
    return signature;
}

void requireMatchingSignature(const std::vector<ModelTensorSignature> &current,
                              const std::vector<ModelTensorSignature> &updated,
                              const std::string &tensorKind) {
    if (current != updated) {
        throw std::invalid_argument("Updated inference model " + tensorKind +
                                    "s do not match the loaded model");
    }
}

PreparedInferenceModel
prepareInferenceModelUpdate(const std::vector<ModelTensorSignature> &parameterSignature,
                            const std::vector<ModelTensorSignature> &bufferSignature,
                            const std::string &modelPath, const torch::Device &device,
                            const torch::Dtype dataType, const at::MemoryFormat memoryFormat,
                            const torch::Tensor &validationInput, const std::int64_t actionCount,
                            const std::int64_t outcomeCount) {
    auto loadedModel = std::make_unique<torch::jit::script::Module>(
        loadInferenceModel(modelPath, device, dataType, memoryFormat));
    requireMatchingSignature(parameterSignature, tensorSignature(loadedModel->named_parameters()),
                             "parameter");
    requireMatchingSignature(bufferSignature, tensorSignature(loadedModel->named_buffers()),
                             "buffer");
    // Freezing folds the weights into constants, which is what makes the graph capture replayable
    // and is a fifth of the host dispatch cost on its own.
    auto updatedModel =
        std::make_unique<torch::jit::script::Module>(torch::jit::freeze(*loadedModel));
    torch::InferenceMode inferenceMode;
    const torch::jit::IValue output = updatedModel->forward({validationInput});
    if (!output.isTuple()) {
        throw std::invalid_argument("Updated inference model must return a tuple");
    }
    const auto outputTuple = output.toTuple();
    if (outputTuple->elements().size() != 3 || !outputTuple->elements()[0].isTensor() ||
        !outputTuple->elements()[1].isTensor() || !outputTuple->elements()[2].isTensor()) {
        throw std::invalid_argument(
            "Updated inference model must return policy, WDL, and search-budget tensors");
    }
    const torch::Tensor policy = outputTuple->elements()[0].toTensor();
    const torch::Tensor outcome = outputTuple->elements()[1].toTensor();
    const torch::Tensor searchBudget = outputTuple->elements()[2].toTensor();
    if (policy.dim() != 2 || policy.size(0) != 1 || policy.size(1) != actionCount ||
        outcome.dim() != 2 || outcome.size(0) != 1 || outcome.size(1) != outcomeCount ||
        !torch::isfinite(policy).all().item<bool>() ||
        !torch::isfinite(outcome).all().item<bool>() || (outcome < 0).any().item<bool>() ||
        searchBudget.dim() != 2 || searchBudget.size(0) != 1 ||
        searchBudget.size(1) != static_cast<std::int64_t>(SEARCH_BUDGET_CURVE_POINTS) ||
        !torch::isfinite(searchBudget).all().item<bool>() ||
        std::abs(outcome.sum().item<float>() - 1.0F) > 1e-2F) {
        throw std::invalid_argument("Updated inference model returned invalid output");
    }
    return updatedModel;
}

void collectConstantTensors(const torch::jit::Block &block, std::vector<torch::Tensor> &tensors) {
    for (const torch::jit::Node *node : block.nodes()) {
        if (node->kind() == torch::jit::prim::Constant &&
            node->hasAttribute(torch::jit::attr::value) &&
            node->kindOf(torch::jit::attr::value) == torch::jit::AttributeKind::t) {
            tensors.push_back(node->t(torch::jit::attr::value));
        }
        for (const torch::jit::Block *nested : node->blocks()) {
            collectConstantTensors(*nested, tensors);
        }
    }
}

// Freezing turns every weight into a graph constant, and a captured graph launches its kernels
// against those constants' addresses; the addresses are the whole reason the weights can be
// replaced without recapturing.
[[nodiscard]] std::vector<torch::Tensor>
frozenConstantTensors(const torch::jit::script::Module &module) {
    std::vector<torch::Tensor> tensors;
    collectConstantTensors(*module.get_method("forward").graph()->block(), tensors);
    return tensors;
}

[[nodiscard]] bool interchangeableConstants(const std::vector<torch::Tensor> &current,
                                            const std::vector<torch::Tensor> &updated) {
    if (current.empty() || current.size() != updated.size()) {
        return false;
    }
    for (const auto index : range(current.size())) {
        if (current[index].sizes() != updated[index].sizes() ||
            current[index].strides() != updated[index].strides() ||
            current[index].dtype() != updated[index].dtype() ||
            current[index].device() != updated[index].device()) {
            return false;
        }
    }
    return true;
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

// CPU inference has no reduced-precision kernels for this graph, so it stays float32 whatever the
// configuration asks for; only CUDA honours the requested precision.
torch::Dtype resolveDtype(const torch::Device &device, const InferencePrecision precision) {
    if (!device.is_cuda()) {
        return torch::kFloat32;
    }
    switch (precision) {
    case InferencePrecision::Float16:
        return torch::kHalf;
    case InferencePrecision::Float32:
        return torch::kFloat32;
    case InferencePrecision::BFloat16:
        break;
    }
    return torch::kBFloat16;
}

at::MemoryFormat resolveMemoryFormat(const torch::Device &device,
                                     const InferenceMemoryFormat memoryFormat) {
    if (!device.is_cuda() || memoryFormat == InferenceMemoryFormat::Contiguous) {
        return at::MemoryFormat::Contiguous;
    }
    return at::MemoryFormat::ChannelsLast;
}

void configureExecution(const torch::Device &device, const InferenceExecutionOptions &options) {
    const SdpaBackend backend = options.sdpa_backend;
    if (!device.is_cuda() && backend != SdpaBackend::Automatic) {
        throw std::invalid_argument("An explicit SDPA backend requires CUDA inference");
    }
    if (!device.is_cuda() &&
        (options.precision != InferencePrecision::BFloat16 ||
         options.memory_format != InferenceMemoryFormat::Contiguous || options.cudnn_benchmark)) {
        throw std::invalid_argument(
            "Inference precision, memory format and cuDNN benchmarking require CUDA inference");
    }
    // Exhaustive algorithm selection costs host time once per captured shape during warm-up and
    // nothing afterwards, because the graph replays the algorithm the warm-up chose.
    at::globalContext().setBenchmarkCuDNN(options.cudnn_benchmark);
    at::globalContext().setSDPUseFlash(backend == SdpaBackend::Automatic ||
                                       backend == SdpaBackend::Flash);
    at::globalContext().setSDPUseMemEfficient(backend == SdpaBackend::Automatic ||
                                              backend == SdpaBackend::MemoryEfficient);
    at::globalContext().setSDPUseMath(backend == SdpaBackend::Automatic ||
                                      backend == SdpaBackend::Math);
    at::globalContext().setSDPUseCuDNN(backend == SdpaBackend::Automatic ||
                                       backend == SdpaBackend::CuDNN);
}

constexpr std::int64_t tensorSize(const std::size_t size) noexcept {
    return static_cast<std::int64_t>(size);
}

#ifdef USE_CUDA
// Capture registers a device-wide allocation filter and replays execute against private memory
// pools, neither of which tolerates a second inference worker doing the same thing at the same
// time in the same process. One graph is submitted at a time; a launch is microseconds.
std::mutex &graphSerializationMutex() {
    static std::mutex mutex;
    return mutex;
}

// Graphs replayed on separate streams within one process execute concurrently and corrupt one
// another; sharing a stream per device makes the device order them, without the host blocking.
at::cuda::CUDAStream sharedGraphStream(const c10::DeviceIndex deviceIndex) {
    static std::mutex mutex;
    static std::map<c10::DeviceIndex, at::cuda::CUDAStream> streams;
    const std::lock_guard<std::mutex> guard(mutex);
    const auto existing = streams.find(deviceIndex);
    if (existing != streams.end()) {
        return existing->second;
    }
    return streams.emplace(deviceIndex, at::cuda::getStreamFromPool(false, deviceIndex))
        .first->second;
}
#endif
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
                                 const InferenceDimensions dimensions,
                                 const InferenceExecutionOptions executionOptions)
    : m_device(resolveDevice(device, deviceId)), m_executionOptions(executionOptions),
      m_torchDtype(resolveDtype(m_device, executionOptions.precision)),
      m_memoryFormat(resolveMemoryFormat(m_device, executionOptions.memory_format)),
      m_maximumBatchSize(maximumBatchSize), m_dimensions(dimensions),
      m_model(std::make_unique<torch::jit::script::Module>(
          loadInferenceModel(modelPath, m_device, m_torchDtype, m_memoryFormat))) {
    configureExecution(m_device, executionOptions);
    if (maximumBatchSize == 0) {
        throw std::invalid_argument("Maximum inference batch size must be positive");
    }
    if (dimensions.channels == 0 || dimensions.rows == 0 || dimensions.columns == 0 ||
        dimensions.actions == 0 || dimensions.outcomes == 0) {
        throw std::invalid_argument("Inference dimensions must be positive");
    }
#ifdef USE_CUDA
    if (m_device.is_cuda() && useDedicatedCudaStream) {
        m_cudaStream = sharedGraphStream(m_device.index());
    }
#else
    if (m_device.is_cuda() && useDedicatedCudaStream) {
        throw std::runtime_error("Dedicated CUDA streams require a CUDA-enabled native build");
    }
#endif
    m_parameterSignature = tensorSignature(m_model->named_parameters());
    m_bufferSignature = tensorSignature(m_model->named_buffers());
    *m_model = torch::jit::freeze(*m_model);
    m_deviceInput = createDeviceInputBuffer();
    m_deviceTypedInput = torch::empty(
        {tensorSize(m_maximumBatchSize), tensorSize(m_dimensions.channels),
         tensorSize(m_dimensions.rows), tensorSize(m_dimensions.columns)},
        torch::TensorOptions().device(m_device).dtype(m_torchDtype).memory_format(m_memoryFormat));
    const torch::TensorOptions stagingOptions =
        torch::TensorOptions().device(m_device).dtype(torch::kFloat32);
    m_deviceOutputStaging = {
        .policies = torch::empty({tensorSize(m_maximumBatchSize), tensorSize(m_dimensions.actions)},
                                 stagingOptions),
        .outcomes = torch::empty(
            {tensorSize(m_maximumBatchSize), tensorSize(m_dimensions.outcomes)}, stagingOptions),
        .search_budgets = torch::empty(
            {tensorSize(m_maximumBatchSize), static_cast<std::int64_t>(SEARCH_BUDGET_CURVE_POINTS)},
            stagingOptions),
    };
    captureBatchGraphs();
}

size_t InferenceRunner::capturedBatchSize(const size_t batchSize) const noexcept {
#ifdef USE_CUDA
    for (const CapturedInferenceGraph &captured : m_batchGraphs) {
        if (captured.batch_size >= batchSize) {
            return captured.batch_size;
        }
    }
#else
    static_cast<void>(batchSize);
#endif
    return 0;
}

void InferenceRunner::runModelToStaging(const size_t batchSize) {
    const std::int64_t rows = tensorSize(batchSize);
    const torch::Tensor typedInput = m_deviceTypedInput.narrow(0, 0, rows);
    typedInput.copy_(m_deviceInput.narrow(0, 0, rows));
    m_modelInputs[0] = typedInput;
    const auto outputTuple = m_model->forward(m_modelInputs).toTuple();
    if (outputTuple->elements().size() != 3) {
        throw std::runtime_error(
            "Inference model must return policy, WDL, and search-budget tensors");
    }
    m_deviceOutputStaging.policies.narrow(0, 0, rows).copy_(outputTuple->elements()[0].toTensor());
    m_deviceOutputStaging.outcomes.narrow(0, 0, rows).copy_(outputTuple->elements()[1].toTensor());
    m_deviceOutputStaging.search_budgets.narrow(0, 0, rows)
        .copy_(outputTuple->elements()[2].toTensor());
}

void InferenceRunner::runEagerModel(const size_t batchSize, InferenceOutput &output) {
    const torch::Tensor typedInput = m_deviceTypedInput.narrow(0, 0, tensorSize(batchSize));
    typedInput.copy_(m_deviceInput.narrow(0, 0, tensorSize(batchSize)));
    m_modelInputs[0] = typedInput;
    const torch::jit::IValue modelOutput = m_model->forward(m_modelInputs);
    const auto outputTuple = modelOutput.toTuple();
    if (outputTuple->elements().size() != 3) {
        throw std::runtime_error(
            "Inference model must return policy, WDL, and search-budget tensors");
    }
    stageOutput(outputTuple->elements()[0].toTensor(), m_deviceOutputStaging.policies,
                output.policies, batchSize);
    stageOutput(outputTuple->elements()[1].toTensor(), m_deviceOutputStaging.outcomes,
                output.outcomes, batchSize);
    stageOutput(outputTuple->elements()[2].toTensor(), m_deviceOutputStaging.search_budgets,
                output.search_budgets, batchSize);
}

void InferenceRunner::copyStagedOutput(const size_t batchSize, InferenceOutput &output) {
    const std::int64_t rows = tensorSize(batchSize);
    output.policies.narrow(0, 0, rows)
        .copy_(m_deviceOutputStaging.policies.narrow(0, 0, rows), usesCuda());
    output.outcomes.narrow(0, 0, rows)
        .copy_(m_deviceOutputStaging.outcomes.narrow(0, 0, rows), usesCuda());
    output.search_budgets.narrow(0, 0, rows)
        .copy_(m_deviceOutputStaging.search_budgets.narrow(0, 0, rows), usesCuda());
}

void InferenceRunner::releaseBatchGraphs() noexcept {
#ifdef USE_CUDA
    try {
        m_batchGraphs.clear();
    } catch (...) {
    }
    // The pool is destroyed with its last graph and cannot be captured into again, so the id must
    // not outlive them.
    m_graphPool.reset();
#endif
}

void InferenceRunner::captureBatchGraphs() {
#ifdef USE_CUDA
    const std::lock_guard<std::mutex> serialized(graphSerializationMutex());
    captureBatchGraphsSerialized();
#endif
}

void InferenceRunner::captureBatchGraphsSerialized() {
#ifdef USE_CUDA
    if (!m_device.is_cuda() || !m_cudaStream.has_value()) {
        releaseBatchGraphs();
        return;
    }
    // Escape hatch: capture is the newest part of this path, so it stays switchable on a node
    // without a rebuild. One captured set costs about 160 MiB of private pool per process.
    if (const char *disabled = std::getenv("ALPHAZERO_DISABLE_INFERENCE_GRAPHS");
        disabled != nullptr && disabled[0] != '\0' && disabled[0] != '0') {
        releaseBatchGraphs();
        return;
    }
    // Finer buckets pad less: at a cap of 320 an average batch of 241 rounds to 280 with eight
    // buckets and 260 with sixteen. Sixteen was only reduced to eight while every refresh stranded
    // a private pool, which is no longer the case.
    constexpr size_t bucketCount = 16;
    constexpr size_t warmupIterations = 16;
    constexpr size_t bucketWarmupIterations = 3;
    const c10::cuda::CUDAGuard deviceGuard(m_device);
    const at::cuda::CUDAStreamGuard streamGuard(*m_cudaStream);
    torch::InferenceMode inferenceMode;
    std::unique_ptr<at::cuda::CUDAGraph> capturing;
    // The graphs being replaced stay alive across the capture: they are what keeps the private pool
    // referenced, and PyTorch refuses to capture into a pool whose last graph is gone, so releasing
    // them first would strand that pool's memory and start a fresh one on every refresh.
    std::vector<CapturedInferenceGraph> replacement;
    try {
        // The profiling executor only specialises after a few runs; capturing before it settles
        // would bake its bailout path into the graph.
        // Deliberately staging-only: a device-to-host copy makes the pinned allocator record an
        // event on this stream, and processing that event later inside a capture invalidates it.
        for (const auto iteration : range(warmupIterations)) {
            static_cast<void>(iteration);
            runModelToStaging(m_maximumBatchSize);
        }
        m_cudaStream->synchronize();
        const at::cuda::MempoolId_t pool =
            m_graphPool.has_value() ? *m_graphPool : at::cuda::graph_pool_handle();
        for (const auto bucket : range<size_t>(1, bucketCount + 1)) {
            const size_t batchSize = std::max<size_t>(1, m_maximumBatchSize * bucket / bucketCount);
            if (!replacement.empty() && replacement.back().batch_size >= batchSize) {
                continue;
            }
            // cuDNN picks an algorithm the first time it sees a shape, which allocates workspace
            // and can synchronise; doing that inside a capture invalidates it.
            for (const auto iteration : range(bucketWarmupIterations)) {
                static_cast<void>(iteration);
                runModelToStaging(batchSize);
            }
            m_cudaStream->synchronize();
            capturing = std::make_unique<at::cuda::CUDAGraph>();
            // Thread-local error mode: the default checks the whole process, and PyTorch's pinned
            // host allocator lazily frees blocks from whichever thread allocates next, which
            // invalidates an otherwise valid capture and poisons the stream.
            capturing->capture_begin(pool, cudaStreamCaptureModeThreadLocal);
            runModelToStaging(batchSize);
            capturing->capture_end();
            replacement.push_back({.batch_size = batchSize, .graph = std::move(capturing)});
        }
        m_cudaStream->synchronize();
        m_batchGraphs.swap(replacement);
        m_graphPool = pool;
        ++m_graphCaptureCount;
        // The superseded graphs share their pool blocks with the replacements now, so they can
        // never be replayed again and are dropped while the replay lock is still held.
        replacement.clear();
    } catch (const std::exception &failure) {
        // An abandoned capture leaves the stream capturing, which corrupts every later launch on
        // it, so the in-flight capture has to be ended before the graphs are discarded.
        if (capturing != nullptr) {
            try {
                capturing->capture_end();
            } catch (...) {
            }
            capturing.reset();
        }
        replacement.clear();
        releaseBatchGraphs();
        std::cerr << "Inference graph capture unavailable: " << failure.what() << std::endl;
    }
#endif
}

void InferenceRunner::stageOutput(const torch::Tensor &modelOutput, torch::Tensor &staging,
                                  torch::Tensor &destination, const size_t batchSize) {
    const torch::Tensor rows = destination.narrow(0, 0, tensorSize(batchSize));
    if (modelOutput.scalar_type() == rows.scalar_type()) {
        rows.copy_(modelOutput, usesCuda());
        return;
    }
    const torch::Tensor cast = staging.narrow(0, 0, tensorSize(batchSize));
    cast.copy_(modelOutput);
    rows.copy_(cast, usesCuda());
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
                        torch::TensorOptions().device(m_device).dtype(torch::kInt8));
}

InferenceOutput InferenceRunner::createOutputBuffer() const {
    const torch::TensorOptions options =
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32).pinned_memory(usesCuda());
    return {
        .policies = torch::empty({tensorSize(m_maximumBatchSize), tensorSize(m_dimensions.actions)},
                                 options),
        .outcomes = torch::empty(
            {tensorSize(m_maximumBatchSize), tensorSize(m_dimensions.outcomes)}, options),
        .search_budgets = torch::empty(
            {tensorSize(m_maximumBatchSize), static_cast<std::int64_t>(SEARCH_BUDGET_CURVE_POINTS)},
            options),
    };
}

void InferenceRunner::forwardInto(const torch::Tensor &encodedBoards, const size_t batchSize,
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
        output.outcomes.size(1) != tensorSize(m_dimensions.outcomes) ||
        output.search_budgets.device().is_cuda() ||
        output.search_budgets.scalar_type() != torch::kFloat32 ||
        output.search_budgets.dim() != 2 ||
        output.search_budgets.size(0) < static_cast<int64_t>(batchSize) ||
        output.search_budgets.size(1) != static_cast<std::int64_t>(SEARCH_BUDGET_CURVE_POINTS)) {
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
        const std::int64_t rows = tensorSize(batchSize);
        m_deviceInput.narrow(0, 0, rows).copy_(encodedBoards.narrow(0, 0, rows), usesCuda());
        const size_t capturedBatch = capturedBatchSize(batchSize);
        if (capturedBatch == 0) {
            runEagerModel(batchSize, output);
        } else {
#ifdef USE_CUDA
            // Rows beyond the real batch carry the previous call's bytes; the network has no
            // cross-sample coupling, so the rows that are read back are bit-identical either way.
            const std::lock_guard<std::mutex> serialized(graphSerializationMutex());
            for (const CapturedInferenceGraph &captured : m_batchGraphs) {
                if (captured.batch_size == capturedBatch) {
                    captured.graph->replay();
                    break;
                }
            }
            copyStagedOutput(batchSize, output);
#endif
        }
        completion.record();
    } catch (...) {
        completion.finishFailedSubmission();
        throw;
    }
}

void InferenceRunner::forwardInto(const torch::Tensor &encodedBoards, const size_t batchSize,
                                  InferenceOutput &output) {
    InferenceCompletion completion(usesCuda());
    forwardInto(encodedBoards, batchSize, output, completion);
    completion.wait();
}

PreparedInferenceModel InferenceRunner::prepareModelRefresh(const std::string &modelPath) const {
    const torch::Tensor validationInput = torch::zeros(
        {1, tensorSize(m_dimensions.channels), tensorSize(m_dimensions.rows),
         tensorSize(m_dimensions.columns)},
        torch::TensorOptions().device(m_device).dtype(m_torchDtype).memory_format(m_memoryFormat));
    return prepareInferenceModelUpdate(
        m_parameterSignature, m_bufferSignature, modelPath, m_device, m_torchDtype, m_memoryFormat,
        validationInput, tensorSize(m_dimensions.actions), tensorSize(m_dimensions.outcomes));
}

#ifdef USE_CUDA
bool InferenceRunner::adoptWeightsInPlace(const torch::jit::script::Module &updatedModel) noexcept {
    try {
        const std::vector<torch::Tensor> current = frozenConstantTensors(*m_model);
        const std::vector<torch::Tensor> updated = frozenConstantTensors(updatedModel);
        if (!interchangeableConstants(current, updated)) {
            return false;
        }
        const c10::cuda::CUDAGuard deviceGuard(m_device);
        const at::cuda::CUDAStreamGuard streamGuard(*m_cudaStream);
        const torch::InferenceMode inferenceMode;
        for (const auto index : range(current.size())) {
            current[index].copy_(updated[index]);
        }
        m_cudaStream->synchronize();
        return true;
    } catch (const std::exception &failure) {
        std::cerr << "In-place inference weight refresh unavailable: " << failure.what()
                  << std::endl;
        return false;
    }
}
#endif

void InferenceRunner::commitModelRefresh(PreparedInferenceModel updatedModel) noexcept {
    assert(m_model != nullptr);
    assert(updatedModel != nullptr);
#ifdef USE_CUDA
    // Swap and recapture under the replay lock: the graphs that still hold the private pool replay
    // the previous weights, so nothing may replay them between the swap and their replacement.
    const std::lock_guard<std::mutex> serialized(graphSerializationMutex());
    // Recapturing is what leaks: every fresh module is a fresh TorchScript executor, and warming it
    // at each captured shape makes the JIT compile a specialisation per shape whose device code
    // the process never gets back. Overwriting the weights the live graphs already point at keeps
    // both the graphs and the executor, so nothing is compiled and nothing is stranded.
    if (!m_batchGraphs.empty() && adoptWeightsInPlace(*updatedModel)) {
        return;
    }
    m_model.swap(updatedModel);
    captureBatchGraphsSerialized();
#else
    m_model.swap(updatedModel);
#endif
}

InferencePipeline::InferencePipeline(const std::string &modelPath, const InferenceDevice device,
                                     const int deviceId, const size_t maximumBatchSize,
                                     const size_t slotCount, const bool useDedicatedCudaStream,
                                     const InferenceDimensions dimensions,
                                     const InferenceExecutionOptions executionOptions)
    : m_runner(modelPath, device, deviceId, maximumBatchSize, useDedicatedCudaStream, dimensions,
               executionOptions) {
    if (slotCount < 2) {
        throw std::invalid_argument("Inference pipeline requires at least two slots");
    }
    m_slots.resize(slotCount);
    for (const auto index : range(slotCount)) {
        auto slot = std::make_unique<Slot>(m_runner.usesCuda());
        slot->input = m_runner.createInputBuffer();
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
        .search_budgets =
            slot.output.search_budgets.narrow(0, 0, static_cast<int64_t>(slot.batchSize)),
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
            m_runner.forwardInto(slot.input, slot.batchSize, slot.output, slot.completion);
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
