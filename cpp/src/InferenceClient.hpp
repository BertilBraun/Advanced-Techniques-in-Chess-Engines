#pragma once

#include "common.hpp"

#include "BoardEncoding.hpp"
#include "InferenceClientTypes.hpp"

#include "util/BlockingQueue.hpp"
#include "util/CollisionCheckedCache.hpp"
#include <ATen/core/TensorBody.h>

/**
 * @brief InferenceClient batches and caches inference requests.
 *        It loads a TorchScript model (exported from Python) for inference.
 *
 * The public interface, inference_batch(), returns a vector of InferenceResult in the same
 * order as the input boards. Internally, requests are enqueued and processed asynchronously
 * through dedicated prepare, model, and resolve stages.
 */
class InferenceClient {
public:
    explicit InferenceClient(const InferenceClientParams &args);

    ~InferenceClient();

    /**
     * Synchronous interface: runs inference on a batch of boards and returns results
     * in the same order as the input.
     */
    [[nodiscard]] std::vector<InferenceResult>
    inferenceBatch(const std::vector<const Board *> &boards);

    [[nodiscard]] InferenceStatistics getStatistics();

private:
    typedef std::pair<std::vector<EncodedMoveScore>, float> CachedInferenceResult;
    typedef std::pair<torch::Tensor, float> ModelInferenceResult;

    using InferenceCache = CollisionCheckedCache<BoardFingerprint, CompressedEncodedBoard,
                                                 CachedInferenceResult, 32, BoardFingerprintHash>;
    using CachedInferenceHandle = InferenceCache::ValueHandle;
    using CachedInferenceProducer = InferenceCache::Producer;

    /**
     * @brief Structure representing a single asynchronous inference request.
     */
    struct InferenceRequest {
        torch::Tensor boardTensor;                  // Encoded board representation.
        std::promise<ModelInferenceResult> promise; // Promise to deliver the result.
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

    /**
     * Loads the TorchScript model from file and moves it to the proper device.
     * Assumes the model file is valid.
     */
    void loadModel(const std::string &modelPath);

    /**
     * Collects requests through the oldest request's deadline and stacks their CPU tensors.
     */
    void prepareWorker();

    /**
     * Runs device transfer and model inference for prepared batches.
     */
    void modelWorker();

    /**
     * Resolves completed CPU outputs into the individual request promises.
     */
    void resolveWorker();

    /**
     * Runs model inference for one pre-stacked CPU input tensor.
     *
     * Moves the input to the device and returns policy and value tensors on CPU in float32.
     */
    [[nodiscard]] std::pair<torch::Tensor, torch::Tensor>
    modelInference(const torch::Tensor &inputTensor);

    // Member variables.
    torch::jit::script::Module m_model;
    torch::Device m_device;
    torch::Dtype m_torchDtype;

    // Fingerprints select buckets; every hit is verified against the exact compressed board.
    InferenceCache m_cache;
    std::atomic_size_t m_totalHits = 0;
    std::atomic_size_t m_totalEvals = 0;
    size_t m_totalModelInferenceCalls = 0;     // Guarded by m_modelStatisticsMutex.
    size_t m_totalModelInferencePositions = 0; // Guarded by m_modelStatisticsMutex.
    std::atomic_size_t m_totalFingerprintCollisions = 0;
    std::vector<size_t> m_modelBatchSizeHistogram;

    static constexpr size_t HANDOFF_QUEUE_CAPACITY = 2;

    BlockingQueue<InferenceRequest> m_requestQueue;
    BlockingQueue<PreparedBatch> m_preparedQueue{HANDOFF_QUEUE_CAPACITY};
    BlockingQueue<CompletedBatch> m_completedQueue{HANDOFF_QUEUE_CAPACITY};

    std::thread m_prepareThread;
    std::thread m_modelThread;
    std::thread m_resolveThread;
    std::mutex m_modelStatisticsMutex;
    InferenceClientParams m_params; // Store the parameters for easy access
};
