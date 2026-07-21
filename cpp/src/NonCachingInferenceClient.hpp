#pragma once

#include "common.hpp"

#include "InferenceClientTypes.hpp"
#include "util/BlockingQueue.hpp"
#include <ATen/core/TensorBody.h>

/**
 * @brief Asynchronous inference client with no cache allocation or cache operations.
 */
class NonCachingInferenceClient {
public:
    explicit NonCachingInferenceClient(const InferenceClientParams &args);
    ~NonCachingInferenceClient();

    [[nodiscard]] std::vector<InferenceResult>
    inferenceBatch(const std::vector<const Board *> &boards);
    [[nodiscard]] InferenceStatistics getStatistics();
    void updateModel(const std::string &modelPath);

private:
    struct ModelInferenceResult {
        torch::Tensor policy;
        WdlPrediction outcome;
    };

    struct InferenceRequest {
        torch::Tensor boardTensor;
        std::promise<ModelInferenceResult> promise;
        std::chrono::steady_clock::time_point enqueuedAt;
    };

    struct PreparedBatch {
        torch::Tensor inputTensor;
        std::vector<std::promise<ModelInferenceResult>> promises;
    };

    struct CompletedBatch {
        std::vector<std::promise<ModelInferenceResult>> promises;
        torch::Tensor policies;
        torch::Tensor values;
        std::exception_ptr exception;
    };

    void prepareWorker();
    void modelWorker();
    void resolveWorker();
    [[nodiscard]] std::pair<torch::Tensor, torch::Tensor>
    modelInference(const torch::Tensor &inputTensor);

    torch::jit::script::Module m_model;
    torch::Device m_device;
    torch::Dtype m_torchDtype;

    std::atomic_size_t m_totalEvals = 0;
    size_t m_totalModelInferenceCalls = 0;
    size_t m_totalModelInferencePositions = 0;
    std::vector<size_t> m_modelBatchSizeHistogram;

    static constexpr size_t HANDOFF_QUEUE_CAPACITY = 2;

    BlockingQueue<InferenceRequest> m_requestQueue;
    BlockingQueue<PreparedBatch> m_preparedQueue{HANDOFF_QUEUE_CAPACITY};
    BlockingQueue<CompletedBatch> m_completedQueue{HANDOFF_QUEUE_CAPACITY};

    std::thread m_prepareThread;
    std::thread m_modelThread;
    std::thread m_resolveThread;
    std::mutex m_modelStatisticsMutex;
    InferenceClientParams m_params;
};
