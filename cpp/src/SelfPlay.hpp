#pragma once

#include "common.hpp"

#include "MoveEncoding.hpp"
#include "Network.hpp"
#include "SelfPlayGame.hpp"
#include "TrainingArgs.hpp"
#include "TrainingStats.hpp"

#include "SelfPlayWriter.hpp"

class SelfPlay {
public:
    SelfPlay(Network &model, const TrainingArgs &args)
        : m_model(model), m_args(args), m_selfPlayWriter(args.batchSize) {
        m_model->eval();
    }

    SelfPlayStats selfPlay() {
        std::vector<SelfPlayGame> selfPlayGames(m_args.numParallelGames, SelfPlayGame());
        SelfPlayStats selfPlayStats;

        while (!selfPlayGames.empty()) {
            expandSelfPlayGames(selfPlayGames);

            for (size_t i = 0; i < selfPlayGames.size(); ++i) {
                auto &game = selfPlayGames[i];

                torch::Tensor actionProbabilities = getActionProbabilities(*game.root);
                game.memory.emplace_back(game.board, actionProbabilities);

                Move move = sampleMove(actionProbabilities, *game.root);

                game.push(move);

                if (game.board.isGameOver()) {
                    // If the game is over, add the training data to the self play memory
                    writeTrainingData(game);

                    selfPlayStats.update(game.memory.size(), getBoardResultScore(game.board));

                    // Remove the game from the list of self play games
                    selfPlayGames[i--] = selfPlayGames.back();
                    selfPlayGames.pop_back();
                }
            }
        }

        return selfPlayStats;
    }

private:
    Network &m_model;
    const TrainingArgs &m_args;
    SelfPlayWriter m_selfPlayWriter;

    void expandSelfPlayGames(std::vector<SelfPlayGame> &selfPlayGames) {
        torch::NoGradGuard no_grad; // Disable gradient calculation equivalent to torch.no_grad()

        torch::Tensor noisePolicy = getPolicyWithNoise(selfPlayGames);

        for (size_t i = 0; i < selfPlayGames.size(); ++i) {
            auto &game = selfPlayGames[i];
            auto moves = filterPolicyThenGetMovesAndProbabilities(noisePolicy[i], game.board);

            game.init(moves);
        }

        for (size_t _ = 0; _ < m_args.numIterationsPerTurn; ++_) {
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
                boards.reserve(expandableSelfPlayGames.size());
                for (size_t idx : expandableSelfPlayGames) {
                    boards.push_back(selfPlayGames[idx].node->board);
                }

                auto [policy, value] = m_model->inference(encodeBoards(boards, m_model->device));

                for (size_t i = 0; i < expandableSelfPlayGames.size(); ++i) {
                    size_t idx = expandableSelfPlayGames[i];
                    AlphaMCTSNode *&node = selfPlayGames[idx].node;

                    auto moves = filterPolicyThenGetMovesAndProbabilities(policy[i], node->board);

                    node->expand(moves);
                    node->backPropagate(value[i].item<float>());
                }
            }
        }
    }

    torch::Tensor getPolicyWithNoise(const std::vector<SelfPlayGame> &selfPlayGames) {
        std::vector<Board> encodedBoards;
        for (const auto &game : selfPlayGames) {
            encodedBoards.push_back(game.board);
        }

        auto [policy, _] = m_model->inference(encodeBoards(encodedBoards, m_model->device));

        // Add dirichlet noise to the policy to encourage exploration
        torch::Tensor dirichletNoise = torch::rand({ACTION_SIZE}, torch::kFloat16);
        dirichletNoise /= dirichletNoise.sum();
        dirichletNoise *= m_args.dirichletAlpha;
        dirichletNoise = dirichletNoise.lerp(torch::rand({ACTION_SIZE}, torch::kFloat16),
                                             m_args.dirichletEpsilon);

        return policy.lerp(dirichletNoise, 1);
    }

    torch::Tensor getActionProbabilities(const AlphaMCTSNode &rootNode) const {
        std::vector<PolicyMove> policyMoves;

        for (const auto &child : rootNode.children) {
            policyMoves.push_back(
                {child.move_to_get_here, child.number_of_visits / rootNode.number_of_visits});
        }

        return encodeMoves(policyMoves, rootNode.board.turn);
    }

    Move sampleMove(const torch::Tensor &actionProbabilities, const AlphaMCTSNode &rootNode) const {
        torch::Tensor temperatureActionProbabilities = actionProbabilities;
        if (rootNode.num_played_moves < 30) {
            // Only use temperature for the first 30 moves, then simply use the action probabilities
            // as they are
            temperatureActionProbabilities = actionProbabilities.pow(1 / m_args.temperature);
        }

        int action = torch::multinomial(temperatureActionProbabilities, 1).item<int>();

        return decodeMove(action, rootNode.board.turn);
    }

    void writeTrainingData(SelfPlayGame &game) {
        float resultScore = getBoardResultScore(game.board);
        Color winner = !game.board.turn;

        for (auto &memory : game.memory) {
            auto encodedBoard = encodeBoard(memory.board, torch::kCPU);
            auto score = (memory.board.turn == winner) ? resultScore : -resultScore;

            if (memory.board.turn == BLACK)
                memory.actionProbabilities =
                    flipActionProbabilitiesVertical(memory.actionProbabilities);

            m_selfPlayWriter.write(encodedBoard, memory.actionProbabilities, score);
        }
    }
};