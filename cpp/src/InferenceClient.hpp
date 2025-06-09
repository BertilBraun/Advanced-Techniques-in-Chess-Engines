#pragma once

#include "common.hpp"

#include "BoardEncoding.hpp"

#include "util/ShardedCache.hpp"
#include <ATen/core/TensorBody.h>

// InferenceResult is defined as a pair: (vector of MoveScore, float value)
typedef std::pair<std::vector<MoveScore>, float> InferenceResult;

struct InferenceStatistics {
    float cacheHitRate = 0.0f;                    // Percentage of cache hits.
    size_t uniquePositions = 0;                   // Number of unique positions in the cache.
    size_t cacheSizeMB = 0;                       // Size of the cache in megabytes.
    std::vector<float> nnOutputValueDistribution; // Distribution of neural network output values.
};

struct InferenceClientParams {
    int device_id;                // GPU device id to use (if available), else CPU.
    std::string currentModelPath; // Path used to resolve the model file.
    int maxBatchSize;             // Maximum number of requests to process in one batch.

    InferenceClientParams(int device_id, std::string currentModelPath, int maxBatchSize)
        : device_id(device_id), currentModelPath(std::move(currentModelPath)),
          maxBatchSize(maxBatchSize) {}
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
    std::vector<InferenceResult> inferenceBatch(const std::vector<const Board *> &boards);

    InferenceStatistics getStatistics();

private:
    typedef std::pair<torch::Tensor, float> ModelInferenceResult;
    /**
     * @brief Structure representing a single asynchronous inference request.
     */
    struct InferenceRequest {
        torch::Tensor boardTensor;                  // Encoded board representation.
        std::promise<ModelInferenceResult> promise; // Promise to deliver the result.
    };

    /**
     * Loads the TorchScript model from file and moves it to the proper device.
     * Assumes the model file is valid.
     */
    void loadModel(const std::string &modelPath);

    /**
     * The dedicated worker thread function.
     *
     * This function waits for requests (or times out after 2 ms) and then processes up
     * to m_maxBatchSize requests at once. After performing batched inference, it sets each
     * request's promise and updates the cache.
     */
    void inferenceWorker();

    /**
     * Runs model inference on a batch of input tensors.
     *
     * Applies softmax to the policies, moves policies and values to CPU (float32),
     * and returns a vector of (policy tensor, value) pairs.
     */
    std::vector<std::pair<torch::Tensor, float>>
    modelInference(const std::vector<torch::Tensor> &boards);

    // Member variables.
    torch::jit::script::Module m_model;
    torch::Device m_device;
    torch::Dtype m_torchDtype;

    // Cache: board -> InferenceResult.
    ShardedCache<CompressedEncodedBoard, InferenceResult, 32, BoardHash> m_cache;
    int m_totalHits;
    int m_totalEvals;

    // Request queue for asynchronous batching.
    std::mutex m_queueMutex;
    std::condition_variable m_queueCV;
    std::queue<InferenceRequest> m_requestQueue;
    bool m_shutdown;

    std::thread m_inferenceThread;
    std::mutex m_modelMutex;
    size_t m_maxBatchSize;
};
