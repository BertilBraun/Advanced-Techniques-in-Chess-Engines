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
        Dataset dataset(m_savePath, m_model->device, 10);

        m_model->train(); // Set model to training mode

        for (size_t iteration = m_startingIteration; iteration < m_args.numIterations;
             ++iteration) {
            std::cout << "Training Iteration " << (iteration + 1) << std::endl;
            TrainingStats trainStats;

            dataset.restart();

            while (dataset.size() < 10000) {
                std::cout << "Waiting for more training data. Current size: " << dataset.size()
                          << "/10000\r" << std::flush;
                std::this_thread::sleep_for(std::chrono::minutes(10));
                dataset.restart();
            }

            size_t numTrainingSamples = dataset.size() * m_args.batchSize;
            std::cout << "Training with " << numTrainingSamples << " memories\n";

            std::cout << "Training started at " << currentDateTime() << "\n";

            for (size_t i = 0; tqdm(i, m_args.numEpochs, "Training"); ++i) {
                trainStats += timeit([&] { return train(dataset); }, "train");
            }

            std::cout << "Training finished at " << currentDateTime() << "\n";

            saveLatestModel(iteration);
            learningStats.update(numTrainingSamples, trainStats);

            // Retain 25% of the memory for the next iteration
            dataset.deleteOldMemories(m_args.retentionRate);

            // evaluateAlphaVsStockfish(modelPath);

            std::cout << "Iteration " << (iteration + 1) << std::endl;

            std::cout << "Train Stats:" << std::endl << trainStats.toString() << std::endl;
            std::cout << "Learning stats:" << std::endl << learningStats.toString() << std::endl;

            std::cout << "Timeit stats:" << std::endl << get_timeit_results() << std::endl;
        }

        std::cout << "Learning finished\n";
    }

private:
    TrainingStats train(Dataset &dataset) {
        TrainingStats trainStats;

        while (dataset.hasNext()) {
            auto [states, policyTargets, valueTargets] = dataset.next();
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

        return trainStats;
    }
};
