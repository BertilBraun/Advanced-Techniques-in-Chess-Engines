#include "InferenceClient.hpp"
#include "util/ShardedCache.hpp"

#include "BoardEncoding.hpp"
#include "MoveEncoding.hpp"

InferenceClient::InferenceClient(const InferenceClientParams &args)
    : m_device(torch::kCPU), m_torchDtype(torch::kFloat32), m_shutdown(false),
      m_params(args) {
    // Initialize the model and other members.
    m_totalHits = 0;
    m_totalEvals = 0;

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
    m_shutdown = true;
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

    m_totalEvals += boards.size();

    // Prepare a futures vector for the inference results.
    // This will be used to wait for the results of the inference requests.
    std::vector<std::pair<size_t, std::future<ModelInferenceResult>>> futures;
    futures.reserve(boards.size());

    for (size_t i : range(boards.size())) {
        // Check if the result is already cached.
        // If so, set the promise and continue.
        if (m_cache.contains(encodedBoards[i])) {
            m_totalHits++;
            continue;
        }

        // Insert a sentinel result to mark it as enqueued.
        m_cache.insertIfNotPresent(encodedBoards[i], kSentinelResult);

        // Create and enqueue a new request.
        InferenceRequest req;
        req.boardTensor = toTensor(encodedBoards[i]);
        futures.emplace_back(i, std::move(req.promise.get_future()));
        {
            std::lock_guard<std::mutex> lock(m_queueMutex);
            m_requestQueue.push(std::move(req));
        }
        m_queueCV.notify_one();
    }

    // Wait for all inference futures to complete.
    for (auto &&[i, future] : futures) {
        const auto [policy, value] = future.get();

        assert(std::abs(value) <= 1.0f + 1e-2f &&
               "InferenceClient::inference_batch: value out of bounds");
        // if any element in policy is negative, or the sum of the policy is not close to 1.0, throw
        // an error.
        assert((policy < 0).any().item<bool>() == false &&
               "InferenceClient::inference_batch: policy contains negative values");
        assert(std::abs(policy.sum().item<float>()) < 1.0f + 1e-2f &&
               "InferenceClient::inference_batch: policy does not sum to 1.0");

        m_cache.insert(encodedBoards[i],
                       {filterPolicyThenGetMovesAndProbabilities(policy, boards[i]), value});
    }

    // Wait for all futures in order.
    std::vector<InferenceResult> results;
    results.reserve(boards.size());

    for (const auto &&[encodedBoard, board] : zip(encodedBoards, boards)) {
        CachedInferenceResult result;
        while (true) {
            if (!m_cache.lookup(encodedBoard, result))
                throw std::runtime_error("InferenceClient::inference_batch: cache lookup failed");
            // Wait until the real result is available.
            if (result != kSentinelResult) {
                break; // We have a valid result.
            }
            // Sleep to avoid busy-waiting.
            std::this_thread::sleep_for(std::chrono::microseconds(10));
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

    if (m_totalEvals == 0 || m_cache.empty()) {
        return stats; // Avoid division by zero.
    }

    stats.cacheHitRate =
        static_cast<float>(m_totalHits) / static_cast<float>(m_totalEvals) * 100.0f;
    stats.uniquePositions = m_cache.size();

    stats.nnOutputValueDistribution.reserve(m_cache.size());
    for (const auto &entry : m_cache) {
        stats.nnOutputValueDistribution.push_back(entry.second.second);
    }

    const size_t sizeInBytes =
        m_cache.size() * (sizeof(CachedInferenceResult) + sizeof(CompressedEncodedBoard));
    stats.cacheSizeMB = sizeInBytes / (1024 * 1024); // Convert to MB

    stats.averageNumberOfPositionsInInferenceCall =
        static_cast<float>(m_totalEvals) / static_cast<float>(m_totalModelInferenceCalls);

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
            m_queueCV.wait_for(
                lock, std::chrono::microseconds(m_params.microsecondsTimeoutInferenceThread),
                [this] { return !m_requestQueue.empty() || m_shutdown; });
            if (m_shutdown && m_requestQueue.empty())
                break;

            // Extract up to m_maxBatchSize requests.
            while (!m_requestQueue.empty() && promises.size() < m_params.maxBatchSize) {
                tensorBatch.push_back(m_requestQueue.front().boardTensor);
                promises.push_back(std::move(m_requestQueue.front().promise));
                m_requestQueue.pop();
            }
        }

        if (!promises.empty()) {
            const std::vector<ModelInferenceResult> inferenceResults = modelInference(tensorBatch);

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

    m_totalModelInferenceCalls++;

    // Stack the input tensors into a single batch tensor.
    // The model expects a 4D tensor: (batch_size, channels, height, width).
    const torch::Tensor inputTensor =
        torch::stack(boards).to(torch::TensorOptions().device(m_device).dtype(m_torchDtype));
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

    policies = policies.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
    values = values.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));

    std::vector<std::pair<torch::Tensor, float>> results;
    results.reserve(boards.size());
    for (int i = 0; i < policies.size(0); ++i) {
        const torch::Tensor policy = policies[i];
        const float value = values[i].item<float>();
        results.emplace_back(policy, value);
    }
    return results;
}
