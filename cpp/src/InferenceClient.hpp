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
    std::atomic_size_t m_totalModelInferenceCalls = 0;
    std::atomic_size_t m_totalModelInferencePositions = 0;
    std::atomic_size_t m_totalFingerprintCollisions = 0;
    std::vector<size_t> m_modelBatchSizeHistogram;

    // Request queue for asynchronous batching.
    std::mutex m_queueMutex;
    std::condition_variable m_queueCV;
    std::queue<InferenceRequest> m_requestQueue;
    bool m_shutdown; // Guarded by m_queueMutex.

    static constexpr size_t HANDOFF_QUEUE_CAPACITY = 2;

    std::mutex m_preparedMutex;
    std::condition_variable m_preparedNotEmptyCV;
    std::condition_variable m_preparedNotFullCV;
    std::queue<PreparedBatch> m_preparedQueue;
    bool m_prepareFinished = false; // Guarded by m_preparedMutex.

    std::mutex m_completedMutex;
    std::condition_variable m_completedNotEmptyCV;
    std::condition_variable m_completedNotFullCV;
    std::queue<CompletedBatch> m_completedQueue;
    bool m_modelFinished = false; // Guarded by m_completedMutex.

    std::thread m_prepareThread;
    std::thread m_modelThread;
    std::thread m_resolveThread;
    std::mutex m_modelMutex;
    InferenceClientParams m_params; // Store the parameters for easy access
};
