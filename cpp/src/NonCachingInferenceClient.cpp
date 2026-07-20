#include "NonCachingInferenceClient.hpp"

#include "BoardEncoding.hpp"
#include "MoveEncoding.hpp"

NonCachingInferenceClient::NonCachingInferenceClient(const InferenceClientParams &args)
    : m_device(torch::kCPU), m_torchDtype(torch::kFloat32),
      m_modelBatchSizeHistogram(static_cast<size_t>(args.maxBatchSize) + 1), m_params(args) {
    if (torch::cuda::is_available()) {
        assert(args.device_id >= 0 && args.device_id < torch::cuda::device_count() &&
               "Invalid device ID for CUDA");
        m_device = torch::Device(torch::kCUDA, args.device_id);
        m_torchDtype = torch::kBFloat16;
    }
    loadModel(args.currentModelPath);

    m_prepareThread = std::thread(&NonCachingInferenceClient::prepareWorker, this);
    m_modelThread = std::thread(&NonCachingInferenceClient::modelWorker, this);
    m_resolveThread = std::thread(&NonCachingInferenceClient::resolveWorker, this);
}

NonCachingInferenceClient::~NonCachingInferenceClient() {
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
NonCachingInferenceClient::inferenceBatch(const std::vector<const Board *> &boards) {
    TIMEIT("NonCachingInferenceClient::inferenceBatch");
    if (boards.empty()) {
        return {};
    }

    m_totalEvals.fetch_add(boards.size(), std::memory_order_relaxed);

    std::vector<std::future<ModelInferenceResult>> futures;
    futures.reserve(boards.size());
    std::vector<InferenceRequest> requests;
    requests.reserve(boards.size());
    const std::chrono::steady_clock::time_point enqueuedAt = std::chrono::steady_clock::now();
    for (const Board *board : boards) {
        InferenceRequest request;
        request.boardTensor = toTensor(encodeBoard(board));
        request.enqueuedAt = enqueuedAt;
        futures.push_back(request.promise.get_future());
        requests.push_back(std::move(request));
    }

    const bool published = m_requestQueue.pushBulk(std::move(requests));
    assert(published);

    std::vector<InferenceResult> results;
    results.reserve(boards.size());
    for (size_t index : range(boards.size())) {
        const auto [policy, value] = futures[index].get();
        assert(std::abs(value) <= 1.0f + 1e-2f &&
               "NonCachingInferenceClient::inferenceBatch: value out of bounds");
        assert((policy < 0).any().item<bool>() == false &&
               "NonCachingInferenceClient::inferenceBatch: policy contains negative values");
        assert(std::abs(policy.sum().item<float>()) < 1.0f + 1e-2f &&
               "NonCachingInferenceClient::inferenceBatch: policy does not sum to 1.0");
        results.emplace_back(filterPolicyThenGetMovesAndProbabilities(policy, boards[index]),
                             value);
    }
    return results;
}

InferenceStatistics NonCachingInferenceClient::getStatistics() {
    InferenceStatistics statistics;
    statistics.evaluations = m_totalEvals.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(m_modelStatisticsMutex);
        statistics.modelInferenceCalls = m_totalModelInferenceCalls;
        statistics.modelInferencePositions = m_totalModelInferencePositions;
        statistics.modelBatchSizeHistogram = m_modelBatchSizeHistogram;
    }
    if (statistics.modelInferenceCalls != 0) {
        statistics.averageNumberOfPositionsInInferenceCall =
            static_cast<float>(statistics.modelInferencePositions) /
            static_cast<float>(statistics.modelInferenceCalls);
    }
    return statistics;
}

void NonCachingInferenceClient::loadModel(const std::string &modelPath) {
    std::string modelPathToLoad = modelPath;
    assert((modelPathToLoad.ends_with(".jit.pt") || modelPathToLoad.ends_with(".pt")) &&
           "Model path must end with '.jit.pt' or '.pt'");
    if (!modelPathToLoad.ends_with(".jit.pt")) {
        modelPathToLoad = modelPathToLoad.substr(0, modelPathToLoad.size() - 3) + ".jit.pt";
    }
    assert(std::filesystem::exists(modelPathToLoad) &&
           ("Model file does not exist: " + modelPathToLoad).c_str());

    m_model = torch::jit::load(modelPathToLoad, m_device);
    m_model.to(m_torchDtype);
    m_model.eval();
}

void NonCachingInferenceClient::prepareWorker() {
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

void NonCachingInferenceClient::modelWorker() {
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

void NonCachingInferenceClient::resolveWorker() {
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
            for (size_t index : range(completedBatch->promises.size())) {
                torch::Tensor policy = completedBatch->policies[static_cast<int64_t>(index)];
                const torch::Tensor values = completedBatch->values[static_cast<int64_t>(index)];
                const float value = values[0].item<float>() - values[2].item<float>();
                completedBatch->promises[index].set_value({std::move(policy), value});
            }
        } catch (...) {
            const std::exception_ptr exception = std::current_exception();
            for (std::promise<ModelInferenceResult> &promise : completedBatch->promises) {
                try {
                    promise.set_exception(exception);
                } catch (const std::future_error &) {
                }
            }
        }
    }
}

std::pair<torch::Tensor, torch::Tensor>
NonCachingInferenceClient::modelInference(const torch::Tensor &inputTensor) {
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
    const torch::jit::IValue output = m_model.forward(inputs);
    const auto outputTuple = output.toTuple();
    torch::Tensor policies = outputTuple->elements()[0].toTensor();
    torch::Tensor values = outputTuple->elements()[1].toTensor();
    policies = policies.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
    values = values.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
    return {std::move(policies), std::move(values)};
}
