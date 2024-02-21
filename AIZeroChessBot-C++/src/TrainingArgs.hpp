#pragma once

#include "common.hpp"

struct TrainingArgs {
    // Number of iterations to run first self-play then train
    size_t numIterations;

    // Number of games to run in parallel for self-play
    size_t numParallelGames;

    // Maximum number of searches in the MCTS algorithm per turn in self-play
    size_t numIterationsPerTurn;

    // Number of epochs to train with one batch of self-play data
    size_t numEpochs;

    // Size of the batch to train with
    size_t batchSize;

    // Sampling temperature for the MCTS algorithm in self-play
    float temperature;

    // Epsilon value for dirichlet noise in self-play for exploration
    float dirichletEpsilon;

    // Alpha value for dirichlet noise in self-play for exploration
    float dirichletAlpha;

    // C parameter for the UCB1 formula in the MCTS algorithm in self-play
    float cParam;

    // Path to save the model to after each iteration
    std::string savePath;

    // The percentage of games to retain for training for the next iteration
    int retentionRate;

    // Constructor with default values
    TrainingArgs(size_t numIterations = 0, size_t numParallelGames = 0,
                 int numIterationsPerTurn = 0, size_t numEpochs = 0, size_t batchSize = 0,
                 float temperature = 0.0f, float dirichletEpsilon = 0.0f,
                 float dirichletAlpha = 0.0f, float cParam = 0.0f, std::string savePath = "",
                 int retentionRate = 0)
        : numIterations(numIterations), numParallelGames(numParallelGames),
          numIterationsPerTurn(numIterationsPerTurn), numEpochs(numEpochs), batchSize(batchSize),
          temperature(temperature), dirichletEpsilon(dirichletEpsilon),
          dirichletAlpha(dirichletAlpha), cParam(cParam), savePath(std::move(savePath)),
          retentionRate(retentionRate) {}
};
