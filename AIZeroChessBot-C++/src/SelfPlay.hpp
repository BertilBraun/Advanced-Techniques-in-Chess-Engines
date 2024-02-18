// This bot is heavily based on the Alpha Zero From Scratch Project by foersterrober
// (https://github.com/foersterrobert/AlphaZeroFromScratch/blob/main/9.AlphaParallel.ipynb)

#pragma once

#include "common.hpp"

#include "MoveEncoding.hpp"
#include "Network.hpp"
#include "SelfPlayGame.hpp"
#include "TrainingArgs.hpp"

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

    std::vector<SelfPlayMemory> selfPlay() {
        std::vector<SelfPlayMemory> selfPlayMemory;
        std::vector<SelfPlayGame> selfPlayGames(m_args.numParallelGames, SelfPlayGame());

        while (!selfPlayGames.empty()) {
            expandSelfPlayGames(selfPlayGames);

            Color currentPlayerTurn = selfPlayGames[0].root->board.turn;

            size_t numRemainingGames = selfPlayGames.size();
            for (int i = ((int) selfPlayGames.size()) - 1; i >= 0; --i) {
                auto &game = selfPlayGames[i];

                torch::Tensor actionProbabilities = getActionProbabilities(*game.root);
                game.memory.emplace_back(game.board, actionProbabilities, currentPlayerTurn);

                Move move = sampleMove(actionProbabilities, *game.root);

                game.board = game.root->board.copy();
                game.board.push(move);

                if (game.board.isGameOver()) {
                    // If the game is over, add the training data to the self play memory
                    extend(selfPlayMemory, getTrainingData(game));

                    // Remove the game from the list of self play games
                    selfPlayGames[i] = selfPlayGames[--numRemainingGames];
                }
            }
            selfPlayGames.resize(numRemainingGames);
        }

        return selfPlayMemory;
    }

private:
    Network &m_model;
    const TrainingArgs &m_args;

    void expandSelfPlayGames(std::vector<SelfPlayGame> &selfPlayGames) {
        torch::NoGradGuard no_grad; // Disable gradient calculation equivalent to torch.no_grad()

        torch::Tensor policy = getPolicyWithNoise(selfPlayGames);

        for (size_t i = 0; i < selfPlayGames.size(); ++i) {
            auto &game = selfPlayGames[i];
            auto moves = filterPolicyThenGetMovesAndProbabilities(policy[i], game.board);

            game.root = AlphaMCTSNode::root(game.board);
            game.root->expand(moves);
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
        // Only use temperature for the first 30 moves, then simply use the action probabilities as
        // they are
        torch::Tensor temperatureActionProbabilities;
        if (rootNode.num_played_moves < 30) {
            temperatureActionProbabilities = actionProbabilities.pow(1 / m_args.temperature);
        } else {
            temperatureActionProbabilities = actionProbabilities;
        }

        int action = torch::multinomial(temperatureActionProbabilities, 1).item<int>();

        return decodeMove(action);
    }
};