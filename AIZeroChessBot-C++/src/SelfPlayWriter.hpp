#pragma once

#include "common.hpp"

#include "MoveEncoding.hpp"
#include "Network.hpp"
#include "SelfPlayGame.hpp"
#include "TrainingArgs.hpp"
#include "TrainingStats.hpp"

struct SelfPlayMemory {
    torch::Tensor state;
    torch::Tensor policyTargets;
    torch::Tensor valueTargets;

    SelfPlayMemory(torch::Tensor state, torch::Tensor policyTargets, float valueTargets)
        : state(std::move(state)), policyTargets(std::move(policyTargets)),
          valueTargets(torch::tensor({valueTargets})) {}
};

class SelfPlayWriter {
public:
    SelfPlayWriter(size_t batchSize) : m_batchSize(batchSize) {}

    bool write(const torch::Tensor &board, const torch::Tensor &policy, float resultScore) {

        // if any of the tensors contain NaN, we skip this sample
        if (torch::isnan(board).any().item<bool>()) {
            log("Warning: NaN detected in encoded board");
            return false;
        }
        if (torch::isnan(policy).any().item<bool>()) {
            log("Warning: NaN detected in policy");
            return false;
        }
        if (std::isnan(resultScore)) {
            log("Warning: NaN detected in value");
            return false;
        }

        auto trainingData = symmetricVariations(board, policy, resultScore);
        extend(m_selfPlayMemoryBatch, trainingData);
        saveTrainingDataBatches();
        return true;
    }

private:
    size_t m_batchSize;
    std::vector<SelfPlayMemory> m_selfPlayMemoryBatch;

    std::vector<SelfPlayMemory> symmetricVariations(const torch::Tensor &board,
                                                    const torch::Tensor &actionProbabilities,
                                                    float result) const {
        std::vector<SelfPlayMemory> variations;

        variations.emplace_back(board, actionProbabilities, result);

        variations.emplace_back(flipBoardHorizontal(board),
                                flipActionProbabilitiesHorizontal(actionProbabilities), result);

        variations.emplace_back(flipBoardVertical(board),
                                flipActionProbabilitiesVertical(actionProbabilities), -result);

        variations.emplace_back(
            flipBoardVertical(flipBoardHorizontal(board)),
            flipActionProbabilitiesVertical(flipActionProbabilitiesHorizontal(actionProbabilities)),
            -result);

        return variations;
    }

    void saveTrainingDataBatches() {
        while (m_selfPlayMemoryBatch.size() >= m_batchSize) {
            std::vector<SelfPlayMemory> batch(m_selfPlayMemoryBatch.begin(),
                                              m_selfPlayMemoryBatch.begin() + m_batchSize);
            m_selfPlayMemoryBatch.erase(m_selfPlayMemoryBatch.begin(),
                                        m_selfPlayMemoryBatch.begin() + m_batchSize);

            saveTrainingDataBatch(batch);
        }
    }

    unsigned long long randomId() const {
        return ((unsigned long long) rand() << 32) | (unsigned long long) rand();
    }

    std::filesystem::path getNewBatchSavePath() const {
        std::filesystem::path savePath = MEMORY_DIR / std::to_string(randomId());

        while (std::filesystem::exists(savePath)) {
            savePath = MEMORY_DIR / std::to_string(randomId());
        }

        // Ensure the directory exists
        std::filesystem::create_directories(savePath);

        return savePath;
    }

    void saveTrainingDataBatch(const std::vector<SelfPlayMemory> &memory) const {
        std::filesystem::path memoryPath = getNewBatchSavePath();

        std::vector<torch::Tensor> states;
        std::vector<torch::Tensor> policyTargets;
        std::vector<torch::Tensor> valueTargets;

        for (const auto &mem : memory) {
            states.push_back(mem.state);
            policyTargets.push_back(mem.policyTargets);
            valueTargets.push_back(mem.valueTargets);
        }

        torch::Tensor statesTensor = torch::stack(states);
        torch::Tensor policyTargetsTensor = torch::stack(policyTargets);
        torch::Tensor valueTargetsTensor = torch::stack(valueTargets);

        try {
            torch::save(statesTensor, (memoryPath / "states.pt").string());
            torch::save(policyTargetsTensor, (memoryPath / "policyTargets.pt").string());
            torch::save(valueTargetsTensor, (memoryPath / "valueTargets.pt").string());
        } catch (const std::exception &e) {
            log("Error saving training data batch:", e.what());
        }
    }
};