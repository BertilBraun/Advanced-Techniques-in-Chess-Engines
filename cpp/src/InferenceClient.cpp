#include "InferenceClient.hpp"
#include "util/ShardedCache.hpp"

#include "BoardEncoding.hpp"
#include "MoveEncoding.hpp"

InferenceClient::InferenceClient(const InferenceClientParams &args)
    : m_device(torch::kCPU), m_torchDtype(torch::kFloat32), m_shutdown(false),
      m_maxBatchSize(args.maxBatchSize) {
    // Initialize the model and other members.
    m_totalHits = 0;
    m_totalEvals = 0;

    // Use GPU if available, else CPU.
    if (torch::cuda::is_available()) {
        m_device = torch::Device(torch::kCUDA, args.device_id);
        m_torchDtype = torch::kFloat16; // Use half precision for inference.
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

    std::set<CompressedEncodedBoard> enqueuedBoards; // Track enqueued boards.
    // Prepare a futures vector for the inference results.
    // This will be used to wait for the results of the inference requests.
    std::vector<std::pair<size_t, std::future<ModelInferenceResult>>> futures;
    futures.reserve(boards.size());

    for (size_t i : range(boards.size())) {
        // Check if the result is already cached.
        // If so, set the promise and continue.
        if (m_cache.contains(encodedBoards[i]) || enqueuedBoards.contains(encodedBoards[i])) {
            m_totalHits++;
            continue;
        }

        // Create and enqueue a new request.
        InferenceRequest req;
        req.boardTensor = toTensor(encodedBoards[i], m_device);
        futures.emplace_back(i, std::move(req.promise.get_future()));
        {
            std::lock_guard<std::mutex> lock(m_queueMutex);
            m_requestQueue.push(std::move(req));
        }
        m_queueCV.notify_one();
        enqueuedBoards.insert(encodedBoards[i]); // Mark this board as enqueued.
    }

    // Wait for all inference futures to complete.
    for (auto &&[i, future] : futures) {
        auto [policy, value] = future.get(); // Make a copy of the result.

        m_cache.insert(encodedBoards[i],
                       {filterPolicyThenGetMovesAndProbabilities(policy, boards[i]), value});
    }

    // Wait for all futures in order.
    std::vector<InferenceResult> results;
    results.reserve(boards.size());

    for (const CompressedEncodedBoard &encodedBoard : encodedBoards) {
        InferenceResult result;
        if (!m_cache.lookup(encodedBoard, result))
            throw std::runtime_error("InferenceClient::inference_batch: cache lookup failed");

        results.push_back(result);
    }

    return results;
}

InferenceStatistics InferenceClient::getStatistics() {
    InferenceStatistics stats;

    if (m_totalEvals == 0 || m_cache.empty()) {
        log("No cache statistics to log.");
        return stats; // Avoid division by zero.
    }

    stats.cacheHitRate = (static_cast<float>(m_totalHits) / m_totalEvals) * 100.0f;
    stats.uniquePositions = m_cache.size();

    stats.nnOutputValueDistribution.reserve(m_cache.size());
    for (const auto &entry : m_cache) {
        stats.nnOutputValueDistribution.push_back(entry.second.second);
    }

    const size_t sizeInBytes = m_cache.size() * (sizeof(InferenceResult) + sizeof(CompressedEncodedBoard));
    stats.cacheSizeMB = sizeInBytes / (1024 * 1024); // Convert to MB

    log("Inference Client stats:");
    log("  cache_hit_rate:", stats.cacheHitRate);
    log("  unique_positions:", stats.uniquePositions);
    log("  cache_size_mb:", stats.cacheSizeMB);

    return stats;
}

void InferenceClient::loadModel(const std::string &modelPath) {
    std::lock_guard<std::mutex> lock(m_modelMutex);

    std::cout << "Model Path:" << modelPath << std::endl;
    m_model = torch::jit::load(modelPath, m_device);
    m_model.to(m_torchDtype); // Use half precision for inference.
    m_model.eval();
}

void InferenceClient::inferenceWorker() {
    std::vector<torch::Tensor> tensorBatch;
    std::vector<std::promise<ModelInferenceResult>> promises;
    tensorBatch.reserve(m_maxBatchSize);
    promises.reserve(m_maxBatchSize);

    while (true) {
        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            m_queueCV.wait_for(lock, std::chrono::microseconds(200),
                               [this] { return !m_requestQueue.empty() || m_shutdown; });
            if (m_shutdown && m_requestQueue.empty())
                break;

            // Extract up to m_maxBatchSize requests.
            while (!m_requestQueue.empty() && promises.size() < m_maxBatchSize) {
                tensorBatch.push_back(m_requestQueue.front().boardTensor);
                promises.push_back(std::move(m_requestQueue.front().promise));
                m_requestQueue.pop();
            }
        }

        if (!promises.empty()) {
            const std::vector<std::pair<torch::Tensor, float>> inferenceResults =
                modelInference(tensorBatch);

            for (auto &&[promise, res] : zip(promises, inferenceResults)) {
                promise.set_value(res); // Set the promise for each request.
            }

            tensorBatch.clear(); // Clear the batch for the next iteration.
            promises.clear();    // Clear the promises for the next iteration.
        }
    }
}

std::vector<std::pair<torch::Tensor, float>>
InferenceClient::modelInference(const std::vector<torch::Tensor> &boards) {
    torch::NoGradGuard noGrad;
    std::unique_lock<std::mutex> lock(m_modelMutex);

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

    policies = torch::softmax(policies, 1);
    policies = policies.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
    values = values.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));

    std::vector<std::pair<torch::Tensor, float>> results;
    results.reserve(boards.size());
    for (int i = 0; i < policies.size(0); ++i) {
        const torch::Tensor policy = policies[i];
        const float value = values[i].item<float>();
        results.push_back(std::make_pair(policy, value));
    }
    return results;
}
