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

class SelfPlay {
public:
    SelfPlay(Network &model, const TrainingArgs &args) : m_model(model), m_args(args) {}

    SelfPlayStats selfPlay() {
        std::vector<SelfPlayGame> selfPlayGames(m_args.numParallelGames, SelfPlayGame());
        SelfPlayStats selfPlayStats;

        while (!selfPlayGames.empty()) {
            expandSelfPlayGames(selfPlayGames);

            Color currentPlayerTurn = selfPlayGames[0].root->board.turn;

            size_t numRemainingGames = selfPlayGames.size();

            for (size_t i = 0; i < selfPlayGames.size(); ++i) {
                auto &game = selfPlayGames[i];

                torch::Tensor actionProbabilities = getActionProbabilities(*game.root);
                game.memory.emplace_back(game.board, actionProbabilities, currentPlayerTurn);

                Move move = sampleMove(actionProbabilities, *game.root);

                game.push(move);

                if (game.board.isGameOver()) {
                    // If the game is over, add the training data to the self play memory
                    extend(m_selfPlayMemoryBatch, getTrainingData(game));

                    // Remove the game from the list of self play games
                    selfPlayGames[i--] = selfPlayGames.back();
                    selfPlayGames.pop_back();

                    selfPlayStats.update(game.root->num_played_moves,
                                         getBoardResultScore(game.board));
                }
            }
            saveTrainingDataBatches();
        }

        return selfPlayStats;
    }

private:
    Network &m_model;
    const TrainingArgs &m_args;
    std::vector<SelfPlayMemory> m_selfPlayMemoryBatch;

    void expandSelfPlayGames(std::vector<SelfPlayGame> &selfPlayGames) {
        torch::NoGradGuard no_grad; // Disable gradient calculation equivalent to torch.no_grad()

        torch::Tensor policy = getPolicyWithNoise(selfPlayGames);

        for (size_t i = 0; i < selfPlayGames.size(); ++i) {
            auto &game = selfPlayGames[i];
            auto moves = filterPolicyThenGetMovesAndProbabilities(policy[i], game.board);

            game.init(moves);
        }

        for (int _ = 0; _ < m_args.numIterationsPerTurn; ++_) {
            for (auto &game : selfPlayGames) {
                game.node = game.getBestChildOrBackPropagate(m_args.cParam);
            }

            std::vector<size_t> expandableSelfPlayGames;
            for (size_t i = 0; i < selfPlayGames.size(); ++i) {
                if (selfPlayGames[i].node != nullptr) {
                    expandableSelfPlayGames.push_back(i);
                }
            }

            if (!expandableSelfPlayGames.empty()) {
                std::vector<Board> boards;
                for (size_t idx : expandableSelfPlayGames) {
                    boards.push_back(selfPlayGames[idx].node->board);
                }

                auto [policy, value] = m_model->inference(encodeBoards(boards).to(m_model->device));

                for (size_t i = 0; i < expandableSelfPlayGames.size(); ++i) {
                    size_t idx = expandableSelfPlayGames[i];
                    auto &node = selfPlayGames[idx].node;

                    auto moves = filterPolicyThenGetMovesAndProbabilities(policy[i], node->board);

                    node->expand(moves);
                    node->backPropagate(value[i].item<float>());
                }
            }
        }
    }

    std::vector<SelfPlayMemory> getTrainingData(SelfPlayGame &game) const {
        std::vector<SelfPlayMemory> selfPlayMemory;

        float result = getBoardResultScore(game.board);

        for (const auto &memory : game.memory) {
            torch::Tensor encodedBoard = encodeBoard(memory.board);

            for (const auto &[board, probabilities] :
                 symmetricVariations(encodedBoard, memory.actionProbabilities)) {
                selfPlayMemory.emplace_back(board, probabilities, result);
            }
        }
        return selfPlayMemory;
    }

    std::vector<std::pair<torch::Tensor, torch::Tensor>>
    symmetricVariations(const torch::Tensor &board,
                        const torch::Tensor &actionProbabilities) const {
        std::vector<std::pair<torch::Tensor, torch::Tensor>> variations;

        variations.emplace_back(board, actionProbabilities);

        variations.emplace_back(
            flipBoardHorizontal(board),
            flipActionProbabilities(actionProbabilities, flipMoveIndexHorizontal));

        variations.emplace_back(
            flipBoardVertical(board),
            flipActionProbabilities(actionProbabilities, flipMoveIndexVertical));

        variations.emplace_back(flipBoardVertical(flipBoardHorizontal(board)),
                                flipActionProbabilities(actionProbabilities, [](int idx) {
                                    return flipMoveIndexVertical(flipMoveIndexHorizontal(idx));
                                }));

        return variations;
    }

    torch::Tensor getPolicyWithNoise(const std::vector<SelfPlayGame> &selfPlayGames) {
        std::vector<Board> encodedBoards;
        for (const auto &game : selfPlayGames) {
            encodedBoards.push_back(game.board);
        }

        auto [policy, value] = m_model->inference(encodeBoards(encodedBoards).to(m_model->device));

        // Add dirichlet noise to the policy to encourage exploration
        torch::Tensor dirichletNoise = torch::rand({ACTION_SIZE}, torch::kFloat32);
        dirichletNoise /= dirichletNoise.sum();
        dirichletNoise *= m_args.dirichletAlpha;
        dirichletNoise = dirichletNoise.lerp(torch::rand({ACTION_SIZE}, torch::kFloat32),
                                             m_args.dirichletEpsilon);

        return policy.lerp(dirichletNoise, 1);
    }

    torch::Tensor getActionProbabilities(const AlphaMCTSNode &rootNode) const {
        torch::Tensor actionProbabilities = torch::zeros({ACTION_SIZE}, torch::kFloat32);

        for (const auto &child : rootNode.children) {
            actionProbabilities[encodeMove(child.move_to_get_here)] = child.number_of_visits;
        }
        actionProbabilities /= actionProbabilities.sum();

        return actionProbabilities;
    }

    Move sampleMove(const torch::Tensor &actionProbabilities, const AlphaMCTSNode &rootNode) const {
        torch::Tensor temperatureActionProbabilities = actionProbabilities;
        if (rootNode.num_played_moves < 30) {
            // Only use temperature for the first 30 moves, then simply use the action probabilities
            // as they are
            temperatureActionProbabilities = actionProbabilities.pow(1 / m_args.temperature);
        }

        int action = torch::multinomial(temperatureActionProbabilities, 1).item<int>();

        return decodeMove(action);
    }

    void saveTrainingDataBatches() {
        while (m_selfPlayMemoryBatch.size() >= m_args.batchSize) {
            std::vector<SelfPlayMemory> batch(m_selfPlayMemoryBatch.begin(),
                                              m_selfPlayMemoryBatch.begin() + m_args.batchSize);
            m_selfPlayMemoryBatch.erase(m_selfPlayMemoryBatch.begin(),
                                        m_selfPlayMemoryBatch.begin() + m_args.batchSize);

            saveTrainingDataBatch(batch);
        }
    }

    void saveTrainingDataBatch(const std::vector<SelfPlayMemory> &memory) const {
        std::filesystem::path memoryPath =
            std::filesystem::path(m_args.savePath) / MEMORY_DIR_NAME / std::to_string(rand());

        while (std::filesystem::exists(memoryPath)) {
            memoryPath =
                std::filesystem::path(m_args.savePath) / MEMORY_DIR_NAME / std::to_string(rand());
        }

        // Ensure the directory exists
        std::filesystem::create_directories(memoryPath);

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
    }
};