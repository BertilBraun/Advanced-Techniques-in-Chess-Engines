#pragma once

#include "common.hpp"

#include "TrainingStats.hpp"

#include "Dataset.hpp"

#include "AlphaZeroBase.hpp"

class AlphaZeroTrainer : AlphaZeroBase {
public:
    AlphaZeroTrainer(Network &model, torch::optim::Optimizer &optimizer, const TrainingArgs &args)
        : AlphaZeroBase(model, args, &optimizer) {}

    void run() {
        LearningStats learningStats;
        Dataset dataset(m_savePath, m_model->device, m_args.batchSize, 20);

        m_model->train(); // Set model to training mode

        for (size_t iteration = m_startingIteration; iteration < m_args.numIterations;
             ++iteration) {
            log("Training Iteration", (iteration + 1));

            dataset.load();

            while (dataset.size() < 10000) {
                logNoNewline("Waiting for more training data. Current size:", dataset.size(),
                             "/10000\r");
                std::this_thread::sleep_for(std::chrono::minutes(10));
                dataset.load();
            }

            size_t numTrainingSamples = dataset.size() * m_args.batchSize;
            log("Training with", numTrainingSamples, "memories");

            log("Training started!");

            TrainingStats trainStats;
            // Accumulate stats for each epoch
            // This should show the learning progress over the number of epochs
            //     before the new data is used
            LearningStats epochStats;
            for (size_t i = 0; tqdm(i, m_args.numEpochs, "Training"); ++i) {
                trainStats += timeit([&] { return train(dataset); }, "train");
                epochStats.update(numTrainingSamples, trainStats);
            }

            log("Training finished!");

            saveLatestModel(iteration);
            learningStats.update(numTrainingSamples, trainStats);

            // Retain 25% of the memory for the next iteration
            dataset.deleteOldMemories(m_args.retentionRate);

            // evaluateAlphaVsStockfish(modelPath);

            log("Iteration", (iteration + 1));

            log("Train Stats:\n", trainStats.toString());
            log("Epoch Stats:\n", epochStats.toString());
            log("Learning stats:\n", learningStats.toString());

            log("Timeit stats:\n", getTimeitResults());
        }

        log("Learning finished");
    }

private:
    TrainingStats train(Dataset &dataset) {
        TrainingStats trainStats;

        while (dataset.hasNext()) {
            auto [states, policyTargets, valueTargets] =
                timeit([&] { return dataset.next(); }, "Dataset Next");
            auto [policy, value] = m_model->forward(states);

            auto policyLoss = torch::nn::functional::cross_entropy(policy, policyTargets);
            auto valueLoss = torch::mse_loss(value, valueTargets);
            auto loss = policyLoss + valueLoss;

            m_optimizer->zero_grad();
            loss.backward();
            m_optimizer->step();

            trainStats.update(policyLoss.item<float>(), valueLoss.item<float>(),
                              loss.item<float>());
        }

        // TODO potentially add learning rate scheduler here

        return trainStats;
    }
};
