#pragma once

#include "common.hpp"

#include "BoardEncoding.hpp"
#include "MoveEncoding.hpp"
#include <mutex>

// InferenceResult is defined as a pair: (vector of MoveScore, float value)
typedef std::pair<std::vector<MoveScore>, float> InferenceResult;

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
    /**
     * @param device_id     GPU device id to use (if available), else CPU.
     * @param savePath      Path used to resolve the model file.
     * @param maxBatchSize  Maximum number of requests to process in one batch.
     */
    InferenceClient(const int device_id, const std::string &currentModelPath,
                    const int maxBatchSize, TensorBoardLogger &logger)
        : m_device(torch::kCPU), m_logger(logger), m_totalHits(0), m_totalEvals(0),
          m_shutdown(false), m_maxBatchSize(maxBatchSize) {
        // Use GPU if available, else CPU.
        if (torch::cuda::is_available()) {
            m_device = torch::Device(torch::kCUDA, device_id);
        }
        loadModel(currentModelPath);

        // Start the worker thread that processes inference requests.
        m_inferenceThread = std::thread(&InferenceClient::inferenceWorker, this);
    }

    ~InferenceClient() {
        m_shutdown = true;
        m_queueCV.notify_all();
        if (m_inferenceThread.joinable()) {
            m_inferenceThread.join();
        }
    }

    /**
     * Synchronous interface: runs inference on a batch of boards and returns results
     * in the same order as the input.
     */
    std::vector<InferenceResult> inference_batch(std::vector<Board> &boards) {
        if (boards.empty()) {
            return std::vector<InferenceResult>();
        }

        TimeItGuard timer("InferenceClient::inference_batch");

        // Encode all boards.
        std::vector<CompressedEncodedBoard> encodedBoards;
        encodedBoards.reserve(boards.size());
        for (const Board &board : boards) {
            encodedBoards.push_back(encodeBoard(board));
        }

        // TODO: If multiple requests for the same board are made, currently all of them will be
        // run through the model. This should be optimized to only run the model once and
        // return the same result for all requests.

        // Prepare a futures vector to preserve input order.
        std::vector<std::future<InferenceResult>> futures(boards.size());
        for (size_t i = 0; i < boards.size(); ++i) {
            const int64 h = hash(encodedBoards[i]);
            {
                std::lock_guard<std::mutex> cacheLock(m_cacheMutex);
                m_totalEvals++;

                // Check if the result is already cached.
                // If so, set the promise and continue.
                auto it = m_inferenceCache.find(h);
                if (it != m_inferenceCache.end()) {
                    std::promise<InferenceResult> p;
                    p.set_value(it->second);
                    m_totalHits++;
                    futures[i] = std::move(p).get_future();
                    continue;
                }
            }
            // Create and enqueue a new request.
            InferenceRequest req;
            req.encodedBoard = encodedBoards[i];
            req.board = boards[i].copy();
            req.hash = h;
            futures[i] = req.promise.get_future();
            {
                std::lock_guard<std::mutex> lock(m_queueMutex);
                m_requestQueue.push(std::move(req));
            }
            m_queueCV.notify_one();
        }

        // Wait for all futures in order.
        std::vector<InferenceResult> results;
        results.reserve(boards.size());
        for (auto &&[board, future] : zip(boards, futures)) {
            InferenceResult res = future.get(); // Make a copy of the result.
            // Filter the policy without en passant moves.
            res.first = filterMovesWithLegalMoves(res.first, board);
            results.push_back(res);
        }
        return results;
    }

    /**
     * Reloads the model from the latest file in the save path.
     * Logs cache statistics and clears the cache.
     * This should be called when the model is updated.
     */
    void updateModel(const std::string &modelPath, int iteration) {
        logCacheStatistics(iteration);
        {
            std::lock_guard<std::mutex> lock(m_cacheMutex);
            m_inferenceCache.clear();
        }
        loadModel(modelPath);
    }

private:
    /**
     * @brief Structure representing a single asynchronous inference request.
     */
    struct InferenceRequest {
        CompressedEncodedBoard encodedBoard;   // Encoded board representation.
        Board board;                           // Original board (for move filtering).
        int64_t hash;                          // Computed board hash.
        std::promise<InferenceResult> promise; // Promise to deliver the result.
    };

    /**
     * Loads the TorchScript model from file and moves it to the proper device.
     * Assumes the model file is valid.
     */
    void loadModel(const std::string &modelPath) {
        std::lock_guard<std::mutex> lock(m_modelMutex);

        m_model = torch::jit::load(modelPath, m_device);
        m_model.eval();
    }

    /**
     * Logs cache statistics.
     */
    void logCacheStatistics(int iteration) {
        std::lock_guard<std::mutex> lock(m_cacheMutex);
        if (m_totalEvals == 0) {
            return; // Avoid division by zero.
        }

        const double cacheHitRate = (static_cast<double>(m_totalHits) / m_totalEvals) * 100.0;
        m_logger.add_scalar("cache/hit_rate", iteration, cacheHitRate);
        m_logger.add_scalar("cache/unique_positions", iteration,
                            static_cast<double>(m_inferenceCache.size()));
        std::vector<float> nnOutputValues;
        nnOutputValues.reserve(m_inferenceCache.size());
        for (const auto &entry : m_inferenceCache) {
            nnOutputValues.push_back(entry.second.second);
        }
        m_logger.add_histogram("nn_output_value_distribution", iteration, nnOutputValues);

        size_t sizeInBytes = 0;
        for (const auto &entry : m_inferenceCache) {
            sizeInBytes += sizeof(entry.first) + sizeof(entry.second);
            sizeInBytes += entry.second.first.size() * sizeof(MoveScore);
        }
        const double sizeInMB = static_cast<double>(sizeInBytes) / (1024 * 1024);
        m_logger.add_scalar("cache/size_mb", iteration, sizeInMB);

        log("Inference Client stats on iteration:", iteration);
        log("  cache_hit_rate:", cacheHitRate);
        log("  unique_positions:", m_inferenceCache.size());
        log("  cache_size_mb:", sizeInMB);
    }

    /**
     * The dedicated worker thread function.
     *
     * This function waits for requests (or times out after 2 ms) and then processes up
     * to m_maxBatchSize requests at once. After performing batched inference, it sets each
     * request's promise and updates the cache.
     */
    void inferenceWorker() {
        while (true) {
            std::vector<InferenceRequest> batch;
            {
                std::unique_lock<std::mutex> lock(m_queueMutex);
                m_queueCV.wait_for(lock, std::chrono::milliseconds(2),
                                   [this] { return !m_requestQueue.empty() || m_shutdown; });
                if (m_shutdown && m_requestQueue.empty())
                    break;

                // Extract up to m_maxBatchSize requests.
                while (!m_requestQueue.empty() && batch.size() < m_maxBatchSize) {
                    batch.push_back(std::move(m_requestQueue.front()));
                    m_requestQueue.pop();
                }
            }

            if (!batch.empty()) {
                // Build the input tensor batch.
                std::vector<torch::Tensor> tensorBatch;
                tensorBatch.reserve(batch.size());
                for (const InferenceRequest &req : batch) {
                    tensorBatch.push_back(toTensor(req.encodedBoard, m_device));
                }

                const std::vector<std::pair<torch::Tensor, float>> inferenceResults =
                    modelInference(tensorBatch);

                // For each request in the batch, process the result, update the cache, and
                // fulfill the promise.
                for (std::size_t i = 0; i < batch.size(); ++i) {
                    const auto &[policy, value] = inferenceResults[i];
                    const InferenceResult res = {
                        filterPolicyWithEnPassantMovesThenGetMovesAndProbabilities(policy,
                                                                                   batch[i].board),
                        value};
                    {
                        std::lock_guard<std::mutex> lock(m_cacheMutex);
                        m_inferenceCache[batch[i].hash] = res;
                    }
                    batch[i].promise.set_value(res);
                }
            }
        }
    }

    /**
     * Runs model inference on a batch of input tensors.
     *
     * Applies softmax to the policies, moves policies and values to CPU (float32),
     * and returns a vector of (policy tensor, value) pairs.
     */
    std::vector<std::pair<torch::Tensor, float>>
    modelInference(const std::vector<torch::Tensor> &boards) {
        TimeItGuard timer("InferenceClient::modelInference");
        torch::NoGradGuard no_grad;
        std::unique_lock<std::mutex> lock(m_modelMutex);

        // Stack the input tensors into a single batch tensor.
        // The model expects a 4D tensor: (batch_size, channels, height, width).
        const torch::Tensor inputTensor = torch::stack(boards).to(m_device).to(torch::kFloat32);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(inputTensor);

        // Run the model and get the output.
        // The output is a tuple of (policies, values).
        // policies: (batch_size, ACTION_SIZE)
        // values: (batch_size, 1)
        const torch::jit::IValue output = m_model.forward(inputs);
        const auto outputTuple = output.toTuple();

        torch::Tensor policies = outputTuple->elements()[0].toTensor();
        torch::Tensor values = outputTuple->elements()[1].toTensor();

        policies = torch::softmax(policies, 1);
        policies = policies.to(torch::kCPU).to(torch::kFloat32);
        values = values.to(torch::kCPU).to(torch::kFloat32);

        std::vector<std::pair<torch::Tensor, float>> results;
        results.reserve(boards.size());
        for (int i = 0; i < policies.size(0); ++i) {
            const torch::Tensor policy = policies[i];
            const float value = values[i].item<float>();
            results.push_back(std::make_pair(policy, value));
        }
        return results;
    }

    // Member variables.
    torch::jit::script::Module m_model;
    torch::Device m_device;
    TensorBoardLogger &m_logger;

    // Cache: board hash -> InferenceResult.
    std::unordered_map<int64_t, InferenceResult> m_inferenceCache;
    int m_totalHits;
    int m_totalEvals;
    std::mutex m_cacheMutex;

    // Request queue for asynchronous batching.
    std::mutex m_queueMutex;
    std::condition_variable m_queueCV;
    std::queue<InferenceRequest> m_requestQueue;
    bool m_shutdown;

    std::thread m_inferenceThread;
    std::mutex m_modelMutex;
    size_t m_maxBatchSize;
};
