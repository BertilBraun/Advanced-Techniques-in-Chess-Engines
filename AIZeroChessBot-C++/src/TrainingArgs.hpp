#pragma once

#include "common.hpp"

struct TrainingArgs {
    // Number of iterations to run first self-play then train
    int numIterations;

    // Number of self-play iterations to run per iteration
    int numSelfPlayIterations;

    // Number of games to run in parallel for self-play
    int numParallelGames;

    // Maximum number of searches in the MCTS algorithm per turn in self-play
    int numIterationsPerTurn;

    // Number of epochs to train with one batch of self-play data
    int numEpochs;

    // Number of separate nodes on the cluster for parallelizing self-play
    int numSeparateNodesOnCluster;

    // Size of the batch to train with
    int batchSize;

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

    // Constructor with default values
    TrainingArgs(int numIterations = 0, int numSelfPlayIterations = 0, int numParallelGames = 0,
                 int numIterationsPerTurn = 0, int numEpochs = 0, int numSeparateNodesOnCluster = 0,
                 int batchSize = 0, float temperature = 0.0f, float dirichletEpsilon = 0.0f,
                 float dirichletAlpha = 0.0f, float cParam = 0.0f, std::string savePath = "")
        : numIterations(numIterations), numSelfPlayIterations(numSelfPlayIterations),
          numParallelGames(numParallelGames), numIterationsPerTurn(numIterationsPerTurn),
          numEpochs(numEpochs), batchSize(batchSize), temperature(temperature),
          dirichletEpsilon(dirichletEpsilon), dirichletAlpha(dirichletAlpha), cParam(cParam),
          savePath(std::move(savePath)), numSeparateNodesOnCluster(numSeparateNodesOnCluster) {}
};
