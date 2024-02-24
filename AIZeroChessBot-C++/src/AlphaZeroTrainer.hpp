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
        Dataset dataset(m_savePath, m_model->device, m_args.batchSize, 10);

        m_model->train(); // Set model to training mode

        for (size_t iteration = m_startingIteration; iteration < m_args.numIterations;
             ++iteration) {
            std::cerr << "Training Iteration " << (iteration + 1) << std::endl;

            dataset.load();

            while (dataset.size() < 10000) {
                std::cerr << "Waiting for more training data. Current size: " << dataset.size()
                          << "/10000\r" << std::flush;
                std::this_thread::sleep_for(std::chrono::minutes(10));
                dataset.load();
            }

            size_t numTrainingSamples = dataset.size() * m_args.batchSize;
            std::cerr << "Training with " << numTrainingSamples << " memories\n";

            std::cerr << "Training started at " << currentDateTime() << "\n";

            TrainingStats trainStats;
            // Accumulate stats for each epoch
            // This should show the learning progress over the number of epochs
            //     before the new data is used
            LearningStats epochStats;
            for (size_t i = 0; tqdm(i, m_args.numEpochs, "Training"); ++i) {
                trainStats += timeit([&] { return train(dataset); }, "train");
                epochStats.update(numTrainingSamples, trainStats);
            }

            std::cerr << "Training finished at " << currentDateTime() << "\n";

            saveLatestModel(iteration);
            learningStats.update(numTrainingSamples, trainStats);

            // Retain 25% of the memory for the next iteration
            dataset.deleteOldMemories(m_args.retentionRate);

            // evaluateAlphaVsStockfish(modelPath);

            std::cerr << "Iteration " << (iteration + 1) << std::endl;

            std::cerr << "Train Stats:" << std::endl << trainStats.toString() << std::endl;
            std::cerr << "Epoch Stats:" << std::endl << epochStats.toString() << std::endl;
            std::cerr << "Learning stats:" << std::endl << learningStats.toString() << std::endl;

            std::cerr << "Timeit stats:" << std::endl << get_timeit_results() << std::endl;
        }

        std::cerr << "Learning finished\n";
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

        // TODO potentially add learning rate scheduler here

        return trainStats;
    }
};
