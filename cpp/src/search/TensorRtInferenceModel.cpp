#include "search/TensorRtInferenceModel.hpp"

#include "search/InferencePipeline.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef ALPHAZERO_USE_TENSORRT
#include <ATen/cuda/CUDAContext.h>
#include <NvInfer.h>
#endif

namespace {
std::vector<char> readEngine(const std::string &enginePath) {
    std::ifstream stream(enginePath, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("Could not open TensorRT engine: " + enginePath);
    }
    const std::streamsize size = stream.tellg();
    if (size <= 0) {
        throw std::runtime_error("TensorRT engine is empty: " + enginePath);
    }
    stream.seekg(0, std::ios::beg);
    std::vector<char> bytes(static_cast<size_t>(size));
    if (!stream.read(bytes.data(), size)) {
        throw std::runtime_error("Could not read TensorRT engine: " + enginePath);
    }
    return bytes;
}
} // namespace

#ifdef ALPHAZERO_USE_TENSORRT
namespace {
class TensorRtLogger final : public nvinfer1::ILogger {
public:
    void log(const Severity severity, const char *message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "TensorRT: " << message << '\n';
        }
    }
};

TensorRtLogger &tensorRtLogger() {
    static TensorRtLogger logger;
    return logger;
}

torch::Dtype torchType(const nvinfer1::DataType type) {
    switch (type) {
    case nvinfer1::DataType::kFLOAT:
        return torch::kFloat32;
    case nvinfer1::DataType::kHALF:
        return torch::kFloat16;
    case nvinfer1::DataType::kINT8:
        return torch::kInt8;
    default:
        throw std::runtime_error("Unsupported TensorRT tensor data type");
    }
}

void requireShape(const nvinfer1::Dims &shape, const std::vector<std::int64_t> &expected,
                  const std::string &name) {
    if (shape.nbDims != static_cast<int>(expected.size())) {
        throw std::runtime_error("TensorRT tensor " + name + " has the wrong rank");
    }
    for (int index = 0; index < shape.nbDims; ++index) {
        if (shape.d[index] != expected[static_cast<size_t>(index)]) {
            throw std::runtime_error("TensorRT tensor " + name + " has the wrong shape");
        }
    }
}
} // namespace

class TensorRtInferenceModel::Implementation {
public:
    Implementation(const std::string &enginePath, const int deviceId, const size_t maximumBatchSize,
                   const InferenceDimensions dimensions)
        : m_device(torch::Device(torch::kCUDA, deviceId)), m_dimensions(dimensions),
          m_engineBytes(readEngine(enginePath)),
          m_runtime(nvinfer1::createInferRuntime(tensorRtLogger())) {
        if (m_runtime == nullptr) {
            throw std::runtime_error("TensorRT runtime creation failed");
        }
        m_engine.reset(
            m_runtime->deserializeCudaEngine(m_engineBytes.data(), m_engineBytes.size()));
        if (m_engine == nullptr) {
            throw std::runtime_error("TensorRT engine deserialization failed");
        }
        m_context.reset(m_engine->createExecutionContext());
        if (m_context == nullptr) {
            throw std::runtime_error("TensorRT execution context creation failed");
        }
        const std::int64_t batch = static_cast<std::int64_t>(maximumBatchSize);
        requireShape(m_engine->getTensorShape("states"),
                     {batch, static_cast<std::int64_t>(dimensions.channels),
                      static_cast<std::int64_t>(dimensions.rows),
                      static_cast<std::int64_t>(dimensions.columns)},
                     "states");
        requireShape(m_engine->getTensorShape("policy_logits"),
                     {batch, static_cast<std::int64_t>(dimensions.actions)}, "policy_logits");
        requireShape(m_engine->getTensorShape("wdl_probabilities"),
                     {batch, static_cast<std::int64_t>(dimensions.outcomes)}, "wdl_probabilities");
        const auto inputOptions = torch::TensorOptions().device(m_device).dtype(
            torchType(m_engine->getTensorDataType("states")));
        m_input = torch::empty({batch, static_cast<std::int64_t>(dimensions.channels),
                                static_cast<std::int64_t>(dimensions.rows),
                                static_cast<std::int64_t>(dimensions.columns)},
                               inputOptions);
        m_policies = torch::empty({batch, static_cast<std::int64_t>(dimensions.actions)},
                                  torch::TensorOptions().device(m_device).dtype(
                                      torchType(m_engine->getTensorDataType("policy_logits"))));
        m_outcomes = torch::empty({batch, static_cast<std::int64_t>(dimensions.outcomes)},
                                  torch::TensorOptions().device(m_device).dtype(
                                      torchType(m_engine->getTensorDataType("wdl_probabilities"))));
        if (!m_context->setTensorAddress("states", m_input.data_ptr()) ||
            !m_context->setTensorAddress("policy_logits", m_policies.data_ptr()) ||
            !m_context->setTensorAddress("wdl_probabilities", m_outcomes.data_ptr())) {
            throw std::runtime_error("TensorRT tensor binding failed");
        }
    }

    void forward(const torch::Tensor &encodedInput, const size_t batchSize,
                 InferenceOutput &output) {
        const std::int64_t rows = static_cast<std::int64_t>(batchSize);
        m_input.narrow(0, 0, rows).copy_(encodedInput.narrow(0, 0, rows));
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(m_device.index()).stream();
        if (!m_context->enqueueV3(stream)) {
            throw std::runtime_error("TensorRT inference enqueue failed");
        }
        output.policies.narrow(0, 0, rows).copy_(m_policies.narrow(0, 0, rows));
        output.outcomes.narrow(0, 0, rows).copy_(m_outcomes.narrow(0, 0, rows));
    }

private:
    struct RuntimeDeleter {
        void operator()(nvinfer1::IRuntime *runtime) const noexcept { delete runtime; }
    };
    struct EngineDeleter {
        void operator()(nvinfer1::ICudaEngine *engine) const noexcept { delete engine; }
    };
    struct ContextDeleter {
        void operator()(nvinfer1::IExecutionContext *context) const noexcept { delete context; }
    };

    torch::Device m_device;
    InferenceDimensions m_dimensions;
    std::vector<char> m_engineBytes;
    std::unique_ptr<nvinfer1::IRuntime, RuntimeDeleter> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine, EngineDeleter> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext, ContextDeleter> m_context;
    torch::Tensor m_input;
    torch::Tensor m_policies;
    torch::Tensor m_outcomes;
};
#else
class TensorRtInferenceModel::Implementation {};
#endif

TensorRtInferenceModel::TensorRtInferenceModel(const std::string &enginePath, const int deviceId,
                                               const size_t maximumBatchSize,
                                               const InferenceDimensions dimensions) {
#ifdef ALPHAZERO_USE_TENSORRT
    m_implementation =
        std::make_unique<Implementation>(enginePath, deviceId, maximumBatchSize, dimensions);
#else
    static_cast<void>(enginePath);
    static_cast<void>(deviceId);
    static_cast<void>(maximumBatchSize);
    static_cast<void>(dimensions);
    throw std::runtime_error("TensorRT backend requires a TensorRT-enabled native build");
#endif
}

TensorRtInferenceModel::~TensorRtInferenceModel() = default;

void TensorRtInferenceModel::forward(const torch::Tensor &encodedInput, const size_t batchSize,
                                     InferenceOutput &output) {
#ifdef ALPHAZERO_USE_TENSORRT
    m_implementation->forward(encodedInput, batchSize, output);
#else
    static_cast<void>(encodedInput);
    static_cast<void>(batchSize);
    static_cast<void>(output);
    throw std::runtime_error("TensorRT backend requires a TensorRT-enabled native build");
#endif
}
