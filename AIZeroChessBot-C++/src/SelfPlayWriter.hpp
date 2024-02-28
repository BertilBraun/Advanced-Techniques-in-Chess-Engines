#pragma once

#include "common.hpp"

#include "MoveEncoding.hpp"
#include "Network.hpp"
#include "SelfPlayGame.hpp"
#include "TrainingArgs.hpp"
#include "TrainingStats.hpp"

#include "Dataset.hpp"

struct SelfPlayMemory {
    torch::Tensor state;
    torch::Tensor policyTargets;
    torch::Tensor valueTargets;
    Color turn;

    SelfPlayMemory(torch::Tensor state, torch::Tensor policyTargets, float valueTargets, Color turn)
        : state(std::move(state)), policyTargets(std::move(policyTargets)),
          valueTargets(torch::tensor({valueTargets})), turn(turn) {}
};

class SelfPlayWriter {
public:
    SelfPlayWriter(size_t batchSize) : m_batchSize(batchSize) {}

    bool write(const torch::Tensor &board, const torch::Tensor &policy, float resultScore,
               Color turn) {

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

        auto trainingData = symmetricVariations(board, policy, resultScore, turn);
        extend(m_selfPlayMemoryBatch, trainingData);
        saveTrainingDataBatches();
        return true;
    }

private:
    size_t m_batchSize;
    std::vector<SelfPlayMemory> m_selfPlayMemoryBatch;

    std::vector<SelfPlayMemory> symmetricVariations(const torch::Tensor &board,
                                                    const torch::Tensor &actionProbabilities,
                                                    float result, Color turn) const {
        std::vector<SelfPlayMemory> variations;

        variations.emplace_back(board, actionProbabilities, result, turn);

        variations.emplace_back(flipBoardHorizontal(board),
                                flipActionProbabilitiesHorizontal(actionProbabilities), result,
                                turn);

        variations.emplace_back(flipBoardVertical(board),
                                flipActionProbabilitiesVertical(actionProbabilities), -result,
                                !turn);

        variations.emplace_back(
            flipBoardVertical(flipBoardHorizontal(board)),
            flipActionProbabilitiesVertical(flipActionProbabilitiesHorizontal(actionProbabilities)),
            -result, !turn);

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

        torch::save(statesTensor, (memoryPath / "states.pt").string());
        torch::save(policyTargetsTensor, (memoryPath / "policyTargets.pt").string());
        torch::save(valueTargetsTensor, (memoryPath / "valueTargets.pt").string());

        auto [state, policy, value] = DataSubset::loadSample(memoryPath);

        log("Analyzing sample:", memoryPath, "with", state.size(0), "positions");

        int64_t batchSize = state.size(0);

        for (int64_t i = 0; i < batchSize; i++) {
            Board board = decodeBoard(state[i]);
            auto moves = __mapPolicyToMoves(policy[i], board.turn);

            // sort moves by probability
            std::sort(moves.begin(), moves.end(),
                      [](const auto &lhs, const auto &rhs) { return lhs.second > rhs.second; });

            std::cout << "-------------------------------------------------" << std::endl;
            std::cout << "Actual Turn: " << COLOR_NAMES[memory[i].turn] << std::endl;
            std::cout << board.unicode(true, true) << std::endl;
            std::cout << "FEN: " << board.fen() << std::endl;
            std::cout << "Evaluation: " << value[i].item<float>() << std::endl;
            std::cout << "Policy: " << moves.size() << " moves" << std::endl;
            for (auto [move, score] : moves) {
                std::cout << move.uci() << " " << score << std::endl;
                std::cout << decodeMove(flipMoveIndexVertical(encodeMove(move, board.turn)),
                                        board.turn)
                                 .uci()
                          << " " << score << std::endl;
            }
            std::cout << "Board with best move applied:" << std::endl;
            board.push(moves[0].first);
            std::cout << board.unicode(true, true) << std::endl;
        }
        std::cout << "-------------------------------------------------" << std::endl;
        std::cout << "-------------------------------------------------" << std::endl;
        std::cout << "Empty Board: " << std::endl;
        std::cout << Board().unicode(true, true) << std::endl;
        for (auto move : Board().legalMoves()) {
            std::cout << move.uci() << std::endl;
        }
    }
};