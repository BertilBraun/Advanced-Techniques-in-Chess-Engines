#include "InferenceClient.hpp"
#include "BoardEncoding.hpp"
#include "InferenceModel.hpp"
#include "MoveEncoding.hpp"

InferenceClient::InferenceClient(const InferenceClientParams &args)
    : m_device(torch::kCPU), m_torchDtype(torch::kFloat32), m_cache(args.cacheCapacity),
      m_modelBatchSizeHistogram(static_cast<size_t>(args.maxBatchSize) + 1), m_params(args) {
    if (args.cacheCapacity == 0) {
        throw std::invalid_argument("cacheCapacity must be positive for cached inference");
    }
    const bool useCuda = args.device == InferenceDevice::Cuda ||
                         (args.device == InferenceDevice::Auto && torch::cuda::is_available());
    if (useCuda) {
        if (!torch::cuda::is_available()) {
            throw std::invalid_argument("CUDA inference requested but CUDA is unavailable");
        }
        if (args.device_id < 0 || args.device_id >= torch::cuda::device_count()) {
            throw std::invalid_argument("Invalid CUDA device ID");
        }
        m_device = torch::Device(torch::kCUDA, args.device_id);
        m_torchDtype = torch::kBFloat16; // Use half precision for inference.
    }
    m_model = loadInferenceModel(args.currentModelPath, m_device, m_torchDtype);

    m_prepareThread = std::thread(&InferenceClient::prepareWorker, this);
    m_modelThread = std::thread(&InferenceClient::modelWorker, this);
    m_resolveThread = std::thread(&InferenceClient::resolveWorker, this);
}

InferenceClient::~InferenceClient() {
    m_requestQueue.close();
    if (m_prepareThread.joinable()) {
        m_prepareThread.join();
    }
    if (m_modelThread.joinable()) {
        m_modelThread.join();
    }
    if (m_resolveThread.joinable()) {
        m_resolveThread.join();
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
        std::optional<CachedInferenceProducer> cacheProducer;
        std::future<ModelInferenceResult> future;
    };
    std::vector<PendingInference> futures;
    futures.reserve(boards.size());
    std::vector<InferenceRequest> requests;
    requests.reserve(boards.size());
    std::vector<CachedInferenceHandle> cachedResults(boards.size());
    std::vector<bool> cacheLeases(boards.size());
    std::vector<std::optional<CachedInferenceResult>> uncachedResults(boards.size());

    for (size_t i : range(boards.size())) {
        InferenceCache::Acquisition acquisition =
            m_cache.acquireOrInsert(fingerprints[i], encodedBoards[i]);
        if (acquisition.status == CacheAcquisition::Hit) {
            m_totalHits.fetch_add(1, std::memory_order_relaxed);
            cachedResults[i] = std::move(acquisition.value);
            cacheLeases[i] = true;
            continue;
        }
        if (acquisition.status == CacheAcquisition::Inserted) {
            cachedResults[i] = std::move(acquisition.value);
            cacheLeases[i] = true;
        } else {
            m_totalFingerprintCollisions.fetch_add(1, std::memory_order_relaxed);
        }

        InferenceRequest request;
        request.boardTensor = toTensor(encodedBoards[i]);
        futures.push_back(PendingInference{i, std::move(acquisition.producer),
                                           std::move(request.promise.get_future())});
        requests.push_back(std::move(request));
    }

    if (!requests.empty()) {
        const std::chrono::steady_clock::time_point enqueuedAt = std::chrono::steady_clock::now();
        for (InferenceRequest &request : requests) {
            request.enqueuedAt = enqueuedAt;
        }
        const bool published = m_requestQueue.pushBulk(std::move(requests));
        assert(published);
    }

    const auto releaseCacheLeases = [this, &fingerprints, &cacheLeases] {
        for (size_t i : range(cacheLeases.size())) {
            if (cacheLeases[i]) {
                m_cache.release(fingerprints[i]);
                cacheLeases[i] = false;
            }
        }
    };

    try {
        // Wait for all inference futures to complete.
        for (PendingInference &pending : futures) {
            const size_t i = pending.boardIndex;
            std::future<ModelInferenceResult> &future = pending.future;
            ModelInferenceResult modelResult = future.get();
            const torch::Tensor &policy = modelResult.policy;
            const float value = modelResult.outcome.expectedValue();

            if (policy.dim() != 1 || !torch::isfinite(policy).all().item<bool>() ||
                (policy < 0).any().item<bool>() ||
                std::abs(policy.sum().item<float>() - 1.0f) > 1e-2f || !std::isfinite(value) ||
                std::abs(value) > 1.0f + 1e-2f) {
                throw std::runtime_error("Inference model returned an invalid policy or value");
            }

            CachedInferenceResult result{
                filterPolicyThenGetMovesAndProbabilities(policy, boards[i]), modelResult.outcome};
            if (pending.cacheProducer.has_value()) {
                pending.cacheProducer->publish(std::move(result));
                pending.cacheProducer.reset();
            } else {
                uncachedResults[i] = std::move(result);
            }
        }

        // Wait for all futures in order.
        std::vector<InferenceResult> results;
        results.reserve(boards.size());

        for (size_t i : range(boards.size())) {
            const Board *board = boards[i];
            const CachedInferenceResult &result = uncachedResults[i].has_value()
                                                      ? uncachedResults[i].value()
                                                      : cachedResults[i].get();
            const std::vector<EncodedMoveScore> &moves = result.moves;

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

            results.push_back({std::move(decodedMovesScores), result.outcome});
            if (cacheLeases[i]) {
                m_cache.release(fingerprints[i]);
                cacheLeases[i] = false;
            }
        }

        return results;
    } catch (...) {
        const std::exception_ptr exception = std::current_exception();
        for (PendingInference &pending : futures) {
            if (pending.cacheProducer.has_value()) {
                pending.cacheProducer->publishException(exception);
                pending.cacheProducer.reset();
            }
        }
        releaseCacheLeases();
        std::rethrow_exception(exception);
    }
}

InferenceStatistics InferenceClient::getStatistics() {
    InferenceStatistics stats;
    stats.cacheCapacity = m_cache.maximumSize();
    stats.cacheEvictions = m_cache.evictionCount();
    stats.cacheFingerprintCollisions = m_totalFingerprintCollisions.load(std::memory_order_relaxed);

    const size_t totalEvals = m_totalEvals.load(std::memory_order_relaxed);
    size_t totalModelInferenceCalls = 0;
    size_t totalModelInferencePositions = 0;
    {
        std::lock_guard<std::mutex> lock(m_modelStatisticsMutex);
        totalModelInferenceCalls = m_totalModelInferenceCalls;
        totalModelInferencePositions = m_totalModelInferencePositions;
        stats.modelBatchSizeHistogram = m_modelBatchSizeHistogram;
    }
    stats.evaluations = totalEvals;
    stats.cacheHits = m_totalHits.load(std::memory_order_relaxed);
    stats.modelInferenceCalls = totalModelInferenceCalls;
    stats.modelInferencePositions = totalModelInferencePositions;
    if (totalEvals == 0 || totalModelInferenceCalls == 0 || m_cache.empty()) {
        return stats; // Avoid division by zero.
    }

    stats.cacheHitRate =
        static_cast<float>(stats.cacheHits) / static_cast<float>(totalEvals) * 100.0f;
    stats.uniquePositions = m_cache.size();

    stats.nnOutputValueDistribution.reserve(m_cache.size());
    size_t dynamicValueBytes = 0;
    m_cache.forEachValue([&stats, &dynamicValueBytes](const CompressedEncodedBoard &,
                                                      const CachedInferenceHandle &resultHandle) {
        if (resultHandle.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            return;
        }
        try {
            const CachedInferenceResult &result = resultHandle.get();
            stats.nnOutputValueDistribution.push_back(result.outcome.expectedValue());
            dynamicValueBytes += result.moves.capacity() * sizeof(EncodedMoveScore);
        } catch (...) {
            // A failed producer is excluded from completed-result statistics.
        }
    });

    const size_t sizeInBytes = m_cache.estimatedStaticSizeBytes() + dynamicValueBytes;
    stats.cacheSizeMB = sizeInBytes / (1024 * 1024); // Convert to MB

    stats.averageNumberOfPositionsInInferenceCall =
        static_cast<float>(totalModelInferencePositions) /
        static_cast<float>(totalModelInferenceCalls);

    return stats;
}

void InferenceClient::updateModel(const std::string &modelPath) {
    updateInferenceModel(m_model, modelPath, m_device, m_torchDtype);
    m_cache.clear();
    m_totalHits.store(0, std::memory_order_relaxed);
    m_totalEvals.store(0, std::memory_order_relaxed);
    m_totalFingerprintCollisions.store(0, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(m_modelStatisticsMutex);
    m_totalModelInferenceCalls = 0;
    m_totalModelInferencePositions = 0;
    std::ranges::fill(m_modelBatchSizeHistogram, 0);
}

void InferenceClient::prepareWorker() {
    while (true) {
        std::optional<std::vector<InferenceRequest>> requests = m_requestQueue.popBatch(
            static_cast<size_t>(m_params.maxBatchSize), [this](const InferenceRequest &request) {
                return request.enqueuedAt +
                       std::chrono::microseconds(m_params.microsecondsTimeoutInferenceThread);
            });
        if (!requests.has_value()) {
            break;
        }

        std::vector<torch::Tensor> tensorBatch;
        std::vector<std::promise<ModelInferenceResult>> promises;
        tensorBatch.reserve(requests->size());
        promises.reserve(requests->size());
        for (InferenceRequest &request : *requests) {
            tensorBatch.push_back(std::move(request.boardTensor));
            promises.push_back(std::move(request.promise));
        }

        assert(promises.size() == tensorBatch.size());
        assert(promises.size() <= static_cast<size_t>(m_params.maxBatchSize));
        PreparedBatch preparedBatch;
        try {
            preparedBatch.inputTensor = torch::stack(tensorBatch);
            preparedBatch.promises = std::move(promises);
        } catch (...) {
            const std::exception_ptr exception = std::current_exception();
            for (std::promise<ModelInferenceResult> &promise : promises) {
                promise.set_exception(exception);
            }
            continue;
        }

        const bool published = m_preparedQueue.push(std::move(preparedBatch));
        assert(published);
    }

    m_preparedQueue.close();
}

void InferenceClient::modelWorker() {
    while (std::optional<PreparedBatch> preparedBatch = m_preparedQueue.pop()) {
        CompletedBatch completedBatch{std::move(preparedBatch->promises), {}, {}, nullptr};
        try {
            std::tie(completedBatch.policies, completedBatch.values) =
                modelInference(preparedBatch->inputTensor);
        } catch (...) {
            completedBatch.exception = std::current_exception();
        }

        const bool published = m_completedQueue.push(std::move(completedBatch));
        assert(published);
    }

    m_completedQueue.close();
}

void InferenceClient::resolveWorker() {
    while (std::optional<CompletedBatch> completedBatch = m_completedQueue.pop()) {
        if (completedBatch->exception != nullptr) {
            for (std::promise<ModelInferenceResult> &promise : completedBatch->promises) {
                promise.set_exception(completedBatch->exception);
            }
            continue;
        }

        try {
            assert(completedBatch->policies.size(0) ==
                   static_cast<int64_t>(completedBatch->promises.size()));
            assert(completedBatch->values.size(0) ==
                   static_cast<int64_t>(completedBatch->promises.size()));

            std::vector<ModelInferenceResult> results;
            results.reserve(completedBatch->promises.size());
            for (size_t i : range(completedBatch->promises.size())) {
                const torch::Tensor policy = completedBatch->policies[static_cast<int64_t>(i)];
                const torch::Tensor values = completedBatch->values[static_cast<int64_t>(i)];
                if (values.dim() != 1 || values.numel() != 3 ||
                    !torch::isfinite(values).all().item<bool>() ||
                    (values < 0).any().item<bool>() ||
                    std::abs(values.sum().item<float>() - 1.0f) > 1e-2f) {
                    throw std::runtime_error(
                        "Inference model WDL output must be three probabilities");
                }
                const WdlPrediction outcome{values[0].item<float>(), values[1].item<float>(),
                                            values[2].item<float>()};
                results.push_back({policy, outcome});
            }

            for (size_t i : range(completedBatch->promises.size())) {
                completedBatch->promises[i].set_value(std::move(results[i]));
            }
        } catch (...) {
            const std::exception_ptr exception = std::current_exception();
            for (std::promise<ModelInferenceResult> &promise : completedBatch->promises) {
                try {
                    promise.set_exception(exception);
                } catch (const std::future_error &) {
                    // Results set before a promise failure remain valid.
                }
            }
        }
    }
}

std::pair<torch::Tensor, torch::Tensor>
InferenceClient::modelInference(const torch::Tensor &inputTensor) {
    torch::NoGradGuard noGrad;

    assert(inputTensor.dim() == 4);
    const size_t batchSize = static_cast<size_t>(inputTensor.size(0));
    {
        std::lock_guard<std::mutex> lock(m_modelStatisticsMutex);
        ++m_totalModelInferenceCalls;
        m_totalModelInferencePositions += batchSize;
        assert(batchSize < m_modelBatchSizeHistogram.size());
        ++m_modelBatchSizeHistogram[batchSize];
    }

    const torch::Tensor deviceInputTensor =
        inputTensor.to(torch::TensorOptions().device(m_device).dtype(m_torchDtype));
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(deviceInputTensor);

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

    return {std::move(policies), std::move(values)};
}
