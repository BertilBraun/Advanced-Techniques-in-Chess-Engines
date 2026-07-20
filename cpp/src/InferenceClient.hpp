#pragma once

#include "common.hpp"

#include "BoardEncoding.hpp"

#include "util/CollisionCheckedCache.hpp"
#include <ATen/core/TensorBody.h>

// InferenceResult is defined as a pair: (vector of MoveScore, float value)
typedef std::pair<std::vector<MoveScore>, float> InferenceResult;

struct InferenceStatistics {
    float cacheHitRate = 0.0f; // Percentage of cache hits.
    size_t evaluations = 0;
    size_t cacheHits = 0;
    size_t uniquePositions = 0;                   // Number of unique positions in the cache.
    size_t cacheSizeMB = 0;                       // Size of the cache in megabytes.
    size_t cacheCapacity = 0;                     // Maximum completed cache entries.
    size_t cacheEvictions = 0;                    // Number of completed entries evicted.
    size_t cacheFingerprintCollisions = 0;        // Exact-board mismatches for equal fingerprints.
    std::vector<float> nnOutputValueDistribution; // Distribution of neural network output values.
    size_t modelInferenceCalls = 0;
    size_t modelInferencePositions = 0;
    std::vector<size_t> modelBatchSizeHistogram;
    float averageNumberOfPositionsInInferenceCall =
        0.0f; // Average number of positions in an inference call.
};

struct InferenceClientParams {
    int device_id;                          // GPU device id to use (if available), else CPU.
    std::string currentModelPath;           // Path used to resolve the model file.
    int maxBatchSize;                       // Maximum number of requests to process in one batch.
    int microsecondsTimeoutInferenceThread; // Timeout for the inference worker thread to wait for
                                            // requests.
    size_t cacheCapacity;                   // Maximum cached positions per inference client.

    InferenceClientParams(int device_id, std::string currentModelPath, int maxBatchSize,
                          int microsecondsTimeoutInferenceThread, size_t cacheCapacity)
        : device_id(device_id), currentModelPath(std::move(currentModelPath)),
          maxBatchSize(maxBatchSize),
          microsecondsTimeoutInferenceThread(microsecondsTimeoutInferenceThread),
          cacheCapacity(cacheCapacity) {}
};

/**
 * @brief InferenceClient batches and caches inference requests.
 *        It loads a TorchScript model (exported from Python) for inference.
 *
 * The public interface, inference_batch(), returns a vector of InferenceResult in the same
 * order as the input boards. Internally, requests are enqueued and processed asynchronously
 * on a dedicated worker thread.
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

    /**
     * Loads the TorchScript model from file and moves it to the proper device.
     * Assumes the model file is valid.
     */
    void loadModel(const std::string &modelPath);

    /**
     * The dedicated worker thread function.
     *
     * This function waits indefinitely for the first request, then collects requests until
     * the maximum batch size or the oldest request's collection deadline is reached. After
     * performing batched inference, it sets each request's promise.
     */
    void inferenceWorker();

    /**
     * Runs model inference on a batch of input tensors.
     *
     * Applies softmax to the policies, moves policies and values to CPU (float32),
     * and returns a vector of (policy tensor, value) pairs.
     */
    [[nodiscard]] std::vector<ModelInferenceResult>
    modelInference(const std::vector<torch::Tensor> &boards);

    // Member variables.
    torch::jit::script::Module m_model;
    torch::Device m_device;
    torch::Dtype m_torchDtype;

    // Fingerprints select buckets; every hit is verified against the exact compressed board.
    InferenceCache m_cache;
    std::atomic_size_t m_totalHits = 0;
    std::atomic_size_t m_totalEvals = 0;
    std::atomic_size_t m_totalModelInferenceCalls = 0;
    std::atomic_size_t m_totalModelInferencePositions = 0;
    std::atomic_size_t m_totalFingerprintCollisions = 0;
    std::vector<size_t> m_modelBatchSizeHistogram;

    // Request queue for asynchronous batching.
    std::mutex m_queueMutex;
    std::condition_variable m_queueCV;
    std::queue<InferenceRequest> m_requestQueue;
    bool m_shutdown; // Guarded by m_queueMutex.

    std::thread m_inferenceThread;
    std::mutex m_modelMutex;
    InferenceClientParams m_params; // Store the parameters for easy access
};
