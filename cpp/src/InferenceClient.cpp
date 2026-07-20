#include "InferenceClient.hpp"
#include "BoardEncoding.hpp"
#include "MoveEncoding.hpp"

InferenceClient::InferenceClient(const InferenceClientParams &args)
    : m_device(torch::kCPU), m_torchDtype(torch::kFloat32), m_cache(args.cacheCapacity),
      m_modelBatchSizeHistogram(static_cast<size_t>(args.maxBatchSize) + 1), m_shutdown(false),
      m_params(args) {
    // Use GPU if available, else CPU.
    if (torch::cuda::is_available()) {
        assert(args.device_id >= 0 && args.device_id < torch::cuda::device_count() &&
               "Invalid device ID for CUDA");
        m_device = torch::Device(torch::kCUDA, args.device_id);
        m_torchDtype = torch::kBFloat16; // Use half precision for inference.
    }
    loadModel(args.currentModelPath);

    // Start the worker thread that processes inference requests.
    m_inferenceThread = std::thread(&InferenceClient::inferenceWorker, this);
}

InferenceClient::~InferenceClient() {
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        m_shutdown = true;
    }
    m_queueCV.notify_all();
    if (m_inferenceThread.joinable()) {
        m_inferenceThread.join();
    }
}

std::vector<InferenceResult>
InferenceClient::inferenceBatch(const std::vector<const Board *> &boards) {
    TIMEIT("InferenceClient::inferenceBatch");

    if (boards.empty()) {
        return {};
    }

    // Encode all boards.
    std::vector<CompressedEncodedBoard> encodedBoards;
    encodedBoards.reserve(boards.size());
    for (const Board *board : boards) {
        encodedBoards.push_back(encodeBoard(board));
    }
    std::vector<BoardFingerprint> fingerprints;
    fingerprints.reserve(encodedBoards.size());
    for (const CompressedEncodedBoard &encodedBoard : encodedBoards) {
        fingerprints.push_back(fingerprintBoard(encodedBoard));
    }

    m_totalEvals.fetch_add(boards.size(), std::memory_order_relaxed);

    // Prepare a futures vector for the inference results.
    // This will be used to wait for the results of the inference requests.
    struct PendingInference {
        size_t boardIndex;
        bool cacheResult;
        std::future<ModelInferenceResult> future;
    };
    std::vector<PendingInference> futures;
    futures.reserve(boards.size());
    std::vector<InferenceRequest> requests;
    requests.reserve(boards.size());
    std::vector<std::optional<CachedInferenceResult>> uncachedResults(boards.size());

    for (size_t i : range(boards.size())) {
        const CacheAcquisition acquisition =
            m_cache.acquireOrInsert(fingerprints[i], encodedBoards[i], kSentinelResult);
        if (acquisition == CacheAcquisition::Hit) {
            m_totalHits.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        const bool cacheResult = acquisition == CacheAcquisition::Inserted;
        if (!cacheResult) {
            m_totalFingerprintCollisions.fetch_add(1, std::memory_order_relaxed);
        }

        InferenceRequest request;
        request.boardTensor = toTensor(encodedBoards[i]);
        futures.push_back(
            PendingInference{i, cacheResult, std::move(request.promise.get_future())});
        requests.push_back(std::move(request));
    }

    if (!requests.empty()) {
        const std::chrono::steady_clock::time_point enqueuedAt = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(m_queueMutex);
            for (InferenceRequest &request : requests) {
                request.enqueuedAt = enqueuedAt;
                m_requestQueue.push(std::move(request));
            }
        }
        m_queueCV.notify_one();
    }

    // Wait for all inference futures to complete.
    for (PendingInference &pending : futures) {
        const size_t i = pending.boardIndex;
        std::future<ModelInferenceResult> &future = pending.future;
        const auto [policy, value] = future.get();

        assert(std::abs(value) <= 1.0f + 1e-2f &&
               "InferenceClient::inference_batch: value out of bounds");
        // if any element in policy is negative, or the sum of the policy is not close to 1.0, throw
        // an error.
        assert((policy < 0).any().item<bool>() == false &&
               "InferenceClient::inference_batch: policy contains negative values");
        assert(std::abs(policy.sum().item<float>()) < 1.0f + 1e-2f &&
               "InferenceClient::inference_batch: policy does not sum to 1.0");

        CachedInferenceResult result{filterPolicyThenGetMovesAndProbabilities(policy, boards[i]),
                                     value};
        if (pending.cacheResult) {
            m_cache.update(fingerprints[i], encodedBoards[i], result);
        } else {
            uncachedResults[i] = std::move(result);
        }
    }

    // Wait for all futures in order.
    std::vector<InferenceResult> results;
    results.reserve(boards.size());

    for (size_t i : range(boards.size())) {
        const BoardFingerprint &fingerprint = fingerprints[i];
        const Board *board = boards[i];
        CachedInferenceResult result;
        if (uncachedResults[i].has_value()) {
            result = std::move(uncachedResults[i].value());
        } else {
            while (true) {
                if (!m_cache.lookup(fingerprint, encodedBoards[i], result)) {
                    throw std::runtime_error(
                        "InferenceClient::inference_batch: cache lookup failed");
                }
                // Wait until the real result is available.
                if (result != kSentinelResult) {
                    break;
                }
                // Sleep to avoid busy-waiting.
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
            m_cache.release(fingerprint);
        }

        const auto &[moves, value] = result;

        // Decode the moves from the cached result using the board.
        std::vector<int> encodedMoves;
        std::vector<float> scores;
        encodedMoves.reserve(moves.size());
        scores.reserve(moves.size());
        for (const auto &[encodedMove, score] : moves) {
            encodedMoves.push_back(encodedMove);
            scores.push_back(score);
        }

        const std::vector<Move> decodedMoves = decodeMoves(encodedMoves, board);

        std::vector<MoveScore> decodedMovesScores;
        decodedMovesScores.reserve(decodedMoves.size());
        for (const auto &&[move, score] : zip(decodedMoves, scores)) {
            decodedMovesScores.emplace_back(move, score);
        }

        results.emplace_back(decodedMovesScores, value);
    }

    return results;
}

InferenceStatistics InferenceClient::getStatistics() {
    InferenceStatistics stats;
    stats.cacheCapacity = m_cache.maximumSize();
    stats.cacheEvictions = m_cache.evictionCount();
    stats.cacheFingerprintCollisions = m_totalFingerprintCollisions.load(std::memory_order_relaxed);

    const size_t totalEvals = m_totalEvals.load(std::memory_order_relaxed);
    const size_t totalModelInferenceCalls =
        m_totalModelInferenceCalls.load(std::memory_order_relaxed);
    const size_t totalModelInferencePositions =
        m_totalModelInferencePositions.load(std::memory_order_relaxed);
    stats.evaluations = totalEvals;
    stats.cacheHits = m_totalHits.load(std::memory_order_relaxed);
    stats.modelInferenceCalls = totalModelInferenceCalls;
    stats.modelInferencePositions = totalModelInferencePositions;
    {
        std::lock_guard<std::mutex> lock(m_modelMutex);
        stats.modelBatchSizeHistogram = m_modelBatchSizeHistogram;
    }
    if (totalEvals == 0 || totalModelInferenceCalls == 0 || m_cache.empty()) {
        return stats; // Avoid division by zero.
    }

    stats.cacheHitRate =
        static_cast<float>(stats.cacheHits) / static_cast<float>(totalEvals) * 100.0f;
    stats.uniquePositions = m_cache.size();

    stats.nnOutputValueDistribution.reserve(m_cache.size());
    size_t dynamicValueBytes = 0;
    m_cache.forEachValue([&stats, &dynamicValueBytes](const CompressedEncodedBoard &,
                                                      const CachedInferenceResult &result) {
        stats.nnOutputValueDistribution.push_back(result.second);
        dynamicValueBytes += result.first.capacity() * sizeof(EncodedMoveScore);
    });

    const size_t sizeInBytes = m_cache.estimatedStaticSizeBytes() + dynamicValueBytes;
    stats.cacheSizeMB = sizeInBytes / (1024 * 1024); // Convert to MB

    stats.averageNumberOfPositionsInInferenceCall =
        static_cast<float>(totalModelInferencePositions) /
        static_cast<float>(totalModelInferenceCalls);

    return stats;
}

void InferenceClient::loadModel(const std::string &modelPath) {
    std::string modelPathToLoad = modelPath;

    // check if model ends with ".jit.pt" or ".pt", if not, throw an error.
    assert((modelPathToLoad.ends_with(".jit.pt") || modelPathToLoad.ends_with(".pt")) &&
           "Model path must end with '.jit.pt' or '.pt'");

    // If it ends with ".pt", change the extension to ".jit.pt".
    if (!modelPathToLoad.ends_with(".jit.pt")) {
        // modelPath ends with ".pt", remove it, and append ".jit.pt" instead
        modelPathToLoad = modelPathToLoad.substr(0, modelPathToLoad.size() - 3) + ".jit.pt";
    }

    // Assert that the model file exists.
    assert(std::filesystem::exists(modelPathToLoad) &&
           ("Model file does not exist: " + modelPathToLoad).c_str());

    std::lock_guard<std::mutex> lock(m_modelMutex);

    m_model = torch::jit::load(modelPathToLoad, m_device);
    m_model.to(m_torchDtype);
    m_model.eval();
}

void InferenceClient::inferenceWorker() {
    std::vector<torch::Tensor> tensorBatch;
    std::vector<std::promise<ModelInferenceResult>> promises;
    tensorBatch.reserve(m_params.maxBatchSize);
    promises.reserve(m_params.maxBatchSize);

    while (true) {
        {
            std::unique_lock lock(m_queueMutex);
            m_queueCV.wait(lock, [this] { return !m_requestQueue.empty() || m_shutdown; });
            if (m_shutdown && m_requestQueue.empty()) {
                break;
            }

            const std::chrono::steady_clock::time_point collectionDeadline =
                m_requestQueue.front().enqueuedAt +
                std::chrono::microseconds(m_params.microsecondsTimeoutInferenceThread);
            m_queueCV.wait_until(lock, collectionDeadline, [this] {
                return m_shutdown ||
                       m_requestQueue.size() >= static_cast<size_t>(m_params.maxBatchSize);
            });

            // Extract up to m_maxBatchSize requests.
            while (!m_requestQueue.empty() &&
                   promises.size() < static_cast<size_t>(m_params.maxBatchSize)) {
                tensorBatch.push_back(std::move(m_requestQueue.front().boardTensor));
                promises.push_back(std::move(m_requestQueue.front().promise));
                m_requestQueue.pop();
            }
        }

        if (!promises.empty()) {
            assert(promises.size() == tensorBatch.size());
            assert(promises.size() <= static_cast<size_t>(m_params.maxBatchSize));
            const std::vector<ModelInferenceResult> inferenceResults = modelInference(tensorBatch);
            assert(inferenceResults.size() == promises.size());

            for (auto &&[promise, res] : zip(promises, inferenceResults)) {
                promise.set_value(res); // Set the promise for each request.
            }

            tensorBatch.clear(); // Clear the batch for the next iteration.
            promises.clear();    // Clear the promises for the next iteration.
        }
    }
}

std::vector<InferenceClient::ModelInferenceResult>
InferenceClient::modelInference(const std::vector<torch::Tensor> &boards) {
    torch::NoGradGuard noGrad;
    std::unique_lock<std::mutex> lock(m_modelMutex);

    m_totalModelInferenceCalls.fetch_add(1, std::memory_order_relaxed);
    m_totalModelInferencePositions.fetch_add(boards.size(), std::memory_order_relaxed);
    assert(boards.size() < m_modelBatchSizeHistogram.size());
    ++m_modelBatchSizeHistogram[boards.size()];

    // Stack the input tensors into a single batch tensor.
    // The model expects a 4D tensor: (batch_size, channels, height, width).
    const torch::Tensor inputTensor =
        torch::stack(boards).to(torch::TensorOptions().device(m_device).dtype(m_torchDtype));
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(inputTensor);

    // Run the model and get the output.
    // The output is a tuple of (policies, values).
    // policies: (batch_size, ACTION_SIZE)
    // values: (batch_size, 3), ordered win/draw/loss
    const torch::jit::IValue output = m_model.forward(inputs);
    const auto outputTuple = output.toTuple();

    torch::Tensor policies = outputTuple->elements()[0].toTensor();
    torch::Tensor values = outputTuple->elements()[1].toTensor();

    policies = policies.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
    values = values.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));

    std::vector<std::pair<torch::Tensor, float>> results;
    results.reserve(boards.size());
    for (int i = 0; i < policies.size(0); ++i) {
        const torch::Tensor policy = policies[i];
        const float value = values[i][0].item<float>() - values[i][2].item<float>();
        results.emplace_back(policy, value);
    }
    return results;
}
