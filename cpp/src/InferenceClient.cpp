#include "InferenceClient.hpp"
#include "util/Time.hpp"

InferenceClient::InferenceClient() : m_device(torch::kCPU), m_shutdown(false), m_maxBatchSize(1) {
    // Initialize the model and other members.
    m_totalHits = 0;
    m_totalEvals = 0;
}

void InferenceClient::init(const int device_id, const std::string &currentModelPath,
                           const int maxBatchSize, TensorBoardLogger *logger) {
    m_device = torch::Device(torch::kCPU);
    m_logger = logger;
    m_totalHits = 0;
    m_totalEvals = 0;
    m_shutdown = false;
    m_maxBatchSize = maxBatchSize;

    // Use GPU if available, else CPU.
    if (torch::cuda::is_available()) {
        m_device = torch::Device(torch::kCUDA, device_id);
    }
    loadModel(currentModelPath);

    // Start the worker thread that processes inference requests.
    m_inferenceThread = std::thread(&InferenceClient::inferenceWorker, this);
}

std::vector<InferenceResult> InferenceClient::inference_batch(std::vector<Board> &boards) {
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
    m_totalEvals += boards.size();

    // Prepare a futures vector to preserve input order.
    std::vector<std::pair<size_t, std::future<ModelInferenceResult>>> futures;
    futures.reserve(boards.size());

    for (size_t i : range(boards.size())) {
        const int64 h = hash(encodedBoards[i]);

        // Check if the result is already cached.
        // If so, set the promise and continue.
        if (m_inferenceCache.contains(h)) {
            m_totalHits++;
            continue;
        }

        // Create and enqueue a new request.
        InferenceRequest req;
        req.boardTensor = toTensor(encodedBoards[i], m_device); // TODO verrrrry slow
        futures.emplace_back(i, std::move(req.promise.get_future()));
        {
            std::lock_guard<std::mutex> lock(m_queueMutex);
            m_requestQueue.push(std::move(req));
        }
        m_queueCV.notify_one();
    }

    // Wait for all inference futures to complete.
    for (auto &&[i, future] : futures) {
        auto [policy, value] = future.get(); // Make a copy of the result.

        m_inferenceCache.insert(
            hash(encodedBoards[i]),
            {filterPolicyWithEnPassantMovesThenGetMovesAndProbabilities(policy, boards[i]), value});
    }

    // Wait for all futures in order.
    std::vector<InferenceResult> results;
    results.reserve(boards.size());

    for (auto &&[board, encodedBoard] : zip(boards, encodedBoards)) {
        InferenceResult res;
        if (!m_inferenceCache.lookup(hash(encodedBoard), res))
            throw std::runtime_error("InferenceClient::inference_batch: cache lookup failed");

        // Filter the policy without en passant moves.
        res.first = filterMovesWithLegalMoves(res.first, board);
        results.push_back(res);
    }

    return results;
}

void InferenceClient::updateModel(const std::string &modelPath, int iteration) {
    logCacheStatistics(iteration);
    m_inferenceCache.clear();
    loadModel(modelPath);
}

void InferenceClient::loadModel(const std::string &modelPath) {
    std::lock_guard<std::mutex> lock(m_modelMutex);

    m_model = torch::jit::load(modelPath, m_device);
    m_model.eval();
}

void InferenceClient::logCacheStatistics(int iteration) {
    if (m_totalEvals == 0 || !m_logger || m_inferenceCache.empty()) {
        return; // Avoid division by zero.
    }

    const double cacheHitRate = (static_cast<double>(m_totalHits) / m_totalEvals) * 100.0;
    m_logger->add_scalar("cache/hit_rate", iteration, cacheHitRate);
    m_logger->add_scalar("cache/unique_positions", iteration,
                         static_cast<double>(m_inferenceCache.size()));
    std::vector<float> nnOutputValues;
    nnOutputValues.reserve(m_inferenceCache.size());
    for (const auto &entry : m_inferenceCache) {
        nnOutputValues.push_back(entry.second.second);
    }
    m_logger->add_histogram("nn_output_value_distribution", iteration, nnOutputValues);

    size_t sizeInBytes = 0;
    for (const auto &entry : m_inferenceCache) {
        sizeInBytes += sizeof(entry.first) + sizeof(entry.second);
        sizeInBytes += entry.second.first.size() * sizeof(MoveScore);
    }
    const double sizeInMB = static_cast<double>(sizeInBytes) / (1024 * 1024);
    m_logger->add_scalar("cache/size_mb", iteration, sizeInMB);

    log("Inference Client stats on iteration:", iteration);
    log("  cache_hit_rate:", cacheHitRate);
    log("  unique_positions:", m_inferenceCache.size());
    log("  cache_size_mb:", sizeInMB);
}

void InferenceClient::inferenceWorker() {
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
                tensorBatch.push_back(req.boardTensor);
            }

            const std::vector<std::pair<torch::Tensor, float>> inferenceResults =
                modelInference(tensorBatch);

            for (auto &&[req, res] : zip(batch, inferenceResults)) {
                req.promise.set_value(res); // Set the promise for each request.
            }
        }
    }
}

std::vector<std::pair<torch::Tensor, float>>
InferenceClient::modelInference(const std::vector<torch::Tensor> &boards) {
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
