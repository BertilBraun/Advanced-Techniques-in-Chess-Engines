#pragma once

#include "common.hpp"

#include "BoardEncoding.hpp"
#include "MoveEncoding.hpp"

typedef std::pair<std::vector<MoveScore>, float> InferenceResult;

// Assume that this function is implemented elsewhere.
// It returns a pair where first is the latest model file path and second is the iteration number.
// TODO : Implement this function.
std::pair<std::string, int> get_latest_iteration_safePath(const std::string &savePath);

/**
 * @brief InferenceClient batches and caches inference requests.
 *        It loads a TorchScript model (exported from Python) for inference.
 */
class InferenceClient {
public:
    /**
     * @param device_id GPU device id to use (if available), else CPU.
     * @param savePath Path used to resolve the model file via model_savePath() or
     * get_latest_iteration_safePath().
     */
    InferenceClient(int device_id, const std::string &savePath)
        : m_savePath(savePath), m_device(torch::kCPU) {
        // Choose device based on CUDA availability.
        if (torch::cuda::is_available()) {
            m_device = torch::Device(torch::kCUDA, device_id);
        }

        // Set the last update check to now.
        m_lastUpdateCheck = std::chrono::steady_clock::now();

        // Initially, load the latest model.
        auto [modelPath, iteration] = get_latest_iteration_safePath(m_savePath);
        m_currentModelPath = modelPath;
        m_currentIteration = iteration;
        loadModel(m_currentModelPath);
    }

    /**
     * Loads the TorchScript model from file and moves it to the proper device.
     * Asserts that the loaded model is not empty.
     */
    void loadModel(const std::string &modelPath) {
        m_model = torch::jit::load(modelPath, m_device);
        m_model.eval();
    }

    /**
     * Runs inference on a batch of boards.
     *
     * Before performing inference, if more than five seconds have passed since the last update
     * check, modelUpdateCheck() is called. This function obtains the latest model file path via
     * get_latest_iteration_safePath(). If that path differs from the current one, it logs cache
     * statistics, clears the cache, resets counters, and loads the new model.
     *
     * @param boards A vector of ChessBoard instances.
     * @return A vector of (move list, value) pairs for each board.
     */
    std::vector<InferenceResult> inference_batch(const std::vector<Board> &boards) {
        modelUpdateCheck();

        if (boards.empty()) {
            return {};
        }

        // Reserve space for encoded boards.
        std::vector<CompressedEncodedBoard> encodedBoards;
        encodedBoards.reserve(boards.size());
        for (const auto &board : boards) {
            encodedBoards.push_back(encodeBoard(board));
        }

        std::vector<torch::Tensor> boardsToInfer;
        std::vector<std::pair<int64_t, Board>> inferenceHashesAndBoards;
        std::unordered_set<int64_t> enqueuedHashes;
        boardsToInfer.reserve(boards.size());
        inferenceHashesAndBoards.reserve(boards.size());
        enqueuedHashes.reserve(boards.size());

        for (size_t i = 0; i < boards.size(); ++i) {
            int64_t hashValue = hash(encodedBoards[i]);
            if (enqueuedHashes.find(hashValue) == enqueuedHashes.end() &&
                m_inferenceCache.find(hashValue) == m_inferenceCache.end()) {
                enqueuedHashes.insert(hashValue);
                auto inferenceBoard = decompress(encodedBoards[i]);
                boardsToInfer.push_back(
                    torch::from_blob(inferenceBoard.data(),
                                     {1, ENCODING_CHANNELS, BOARD_LENGTH, BOARD_LENGTH},
                                     torch::kByte)
                        .to(m_device)
                        .to(torch::kFloat32));
                inferenceHashesAndBoards.push_back({hashValue, boards[i]});
            } else {
                ++m_totalHits;
            }
        }
        m_totalEvals += boards.size();

        if (!boardsToInfer.empty()) {
            auto results = modelInference(boardsToInfer);
            // Reserve capacity for the cache insertion if possible.
            for (size_t i = 0; i < inferenceHashesAndBoards.size(); ++i) {
                auto [hash, board] = inferenceHashesAndBoards[i];
                const auto [moves, value] = results[i];
                // Filter the policy with en passant moves and get the moves and probabilities.
                auto filteredMoves =
                    filterPolicyWithEnPassantMovesThenGetMovesAndProbabilities(moves, board);
                // Insert the result into the cache.
                m_inferenceCache[hash] = {filteredMoves, value};
            }
        }

        std::vector<InferenceResult> responses;
        responses.reserve(boards.size());
        // Gather responses from the cache.
        for (CompressedEncodedBoard board : encodedBoards) {
            auto it = m_inferenceCache.find(hash(board));
            // We expect every board hash to be in the cache.
            assert(it != m_inferenceCache.end());
            responses.push_back(it->second);
        }
        return responses;
    }

private:
    /**
     * Checks if more than five seconds have passed since the last model update check.
     * If so, obtains the latest model path and iteration via get_latest_iteration_safePath().
     * If the latest model is different from the current one, logs cache statistics,
     * clears the cache, resets counters, and loads the new model.
     * Finally, updates the last update check time.
     */
    void modelUpdateCheck() {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - m_lastUpdateCheck).count() < 5)
            return;

        auto [modelPath, iteration] = get_latest_iteration_safePath(m_savePath);
        if (modelPath != m_currentModelPath) {
            logCacheStatistics();
            m_inferenceCache.clear();
            m_totalHits = 0;
            m_totalEvals = 0;
            loadModel(modelPath);
            m_currentModelPath = modelPath;
            m_currentIteration = iteration;
        }
        m_lastUpdateCheck = now;
    }

    /**
     * Logs cache statistics.
     */
    void logCacheStatistics() {
        double cacheHitRate = (static_cast<double>(m_totalHits) / m_totalEvals) * 100.0;
        log_scalar("cache/hit_rate", cacheHitRate, m_currentIteration);
        log_scalar("cache/unique_positions", static_cast<double>(m_inferenceCache.size()),
                   m_currentIteration);

        std::vector<float> nnOutputValues;
        nnOutputValues.reserve(m_inferenceCache.size());
        for (const auto &[hash, cacheEntry] : m_inferenceCache) {
            nnOutputValues.push_back(cacheEntry.second);
        }
        log_histogram("nn_output_value_distribution", nnOutputValues, m_currentIteration);

        size_t sizeInBytes = 0;
        for (const auto &[hash, cacheEntry] : m_inferenceCache) {
            sizeInBytes += sizeof(hash) + sizeof(cacheEntry);
            sizeInBytes += cacheEntry.first.size() * sizeof(MoveScore);
        }
        double sizeInMB = static_cast<double>(sizeInBytes) / (1024 * 1024);
        log_scalar("cache/size_mb", sizeInMB, m_currentIteration);
    }

    /**
     * Runs model inference on a batch of encoded boards.
     * Returns a vector of (policy tensor, value) pairs.
     *
     * This version applies softmax on the policies, then moves policies and values separately
     * to the CPU while converting them to float32.
     */
    std::vector<std::pair<torch::Tensor, float>>
    modelInference(const std::vector<torch::Tensor> &boards) {
        torch::NoGradGuard no_grad;

        // Stack the boards into one input tensor.
        torch::Tensor inputTensor = torch::stack(boards).to(m_device);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(inputTensor);
        auto output = m_model.forward(inputs);

        auto outputTuple = output.toTuple();
        torch::Tensor policies = outputTuple->elements()[0].toTensor();
        torch::Tensor values = outputTuple->elements()[1].toTensor();

        policies = torch::softmax(policies, 1);
        policies = policies.to(torch::kCPU).to(torch::kFloat32);
        values = values.to(torch::kCPU).to(torch::kFloat32);

        std::vector<std::pair<torch::Tensor, float>> results;
        results.reserve(boards.size());
        for (int i = 0; i < policies.size(0); ++i) {
            torch::Tensor policy = policies[i];
            float value = values[i].item<float>();
            results.push_back({policy, value});
        }
        return results;
    }

    // Member variables.
    std::string m_savePath;
    torch::jit::script::Module m_model;
    torch::Device m_device;

    std::unordered_map<int64_t, InferenceResult> m_inferenceCache;
    int m_totalHits = 0;
    int m_totalEvals = 0;

    std::chrono::steady_clock::time_point m_lastUpdateCheck;
    std::string m_currentModelPath;
    int m_currentIteration = 0;
};
