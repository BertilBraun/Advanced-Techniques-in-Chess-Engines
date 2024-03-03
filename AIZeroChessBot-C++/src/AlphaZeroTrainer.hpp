#pragma once

#include "common.hpp"

#include "AlphaZeroBase.hpp"
#include "Barrier.hpp"
#include "Dataset.hpp"
#include "TrainingStats.hpp"

class AlphaZeroTrainer : AlphaZeroBase {
public:
    AlphaZeroTrainer(Network &model, torch::optim::Optimizer &optimizer, const TrainingArgs &args)
        : AlphaZeroBase(model, args, &optimizer) {

        // get the available devices
        auto devices = torch::cuda::device_count();
        log("CUDA devices available:", devices);

        if (devices == 0) {
            m_devicesList.push_back(torch::Device(torch::kCPU));
        }

        for (size_t i = 0; i < std::min(5 * devices, args.numTrainers); i++) {
            m_devicesList.push_back(torch::Device(torch::kCUDA, i % devices));
        }

        for (auto &device : m_devicesList) {
            Network modelClone(device);
            synchronizeModel(modelClone);
            m_models.push_back(modelClone);
        }

        initializeAggregateMutexes();
    }

    void run() {
        LearningStats learningStats;
        Dataset dataset(20);

        m_model->train(); // Set model to training mode

        for (size_t iteration = m_startingIteration; iteration < m_args.numIterations;
             ++iteration) {
            log("Training Iteration", (iteration + 1));

            timeit([&] { dataset.load(); }, "load Dataset");

            while (tqdm(dataset.size(), 10000, "Waiting for more training data...")) {
                std::this_thread::sleep_for(std::chrono::minutes(10));
                timeit([&] { dataset.load(); }, "load Dataset");
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
                TrainingStats epochTrainStats = timeit([&] { return train(dataset); }, "train");
                epochStats.update(numTrainingSamples, epochTrainStats);
                trainStats += epochTrainStats;
            }

            log("Training finished!");

            saveLatestModel(iteration);
            learningStats.update(numTrainingSamples, trainStats);

            // Retain 25% of the memory for the next iteration
            timeit([&] { dataset.deleteOldMemories(m_args.retentionRate); }, "deleteOldMemories");

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
    std::vector<torch::Device> m_devicesList;
    std::vector<Network> m_models;
    Barrier m_barrier;

    TrainingStats train(Dataset &dataset) {
        TrainingStats totalTrainStats;
        std::mutex statsMutex;

        std::vector<DataSubset> dataSubsets = dataset.intoDataSubsets(m_devicesList);

        m_barrier.updateRequired(m_devicesList.size());

        // spin up a thread for each device
        std::vector<std::thread> threads;

        for (size_t i = 0; i < m_devicesList.size(); ++i) {
            threads.emplace_back(
                std::thread([i, &dataSubsets, &totalTrainStats, &statsMutex, this] {
                    TrainingStats trainStats =
                        timeit([&] { return trainOnDevice(dataSubsets[i], m_models[i]); },
                               "trainOnDevice");

                    log("Device", i, "finished training");

                    statsMutex.lock();
                    totalTrainStats += trainStats;
                    statsMutex.unlock();
                }));
        }

        for (auto &thread : threads) {
            thread.join();
        }

        // TODO potentially add learning rate scheduler here

        return totalTrainStats;
    }

    TrainingStats trainOnDevice(DataSubset &dataSubset, Network &model) {
        TrainingStats trainStats;

        while (dataSubset.hasNext()) {
            auto [states, policyTargets, valueTargets] = dataSubset.next();
            auto [policy, value] = model->forward(states);

            auto policyLoss = torch::nn::functional::cross_entropy(policy, policyTargets);
            // Scale the values since the value targets are in the range [-1, 1] and mse
            // loss decreases the loss by squaring it when the |value - valueTarget| is less
            // than 1 (which is often the case with values in the range [-1, 1] and values
            // very close to 0). This potentially leads to not learning the value function
            // well -> scale the values to be in the range [-25, 25] to increase the loss
            // and make the learning more stable.
            auto valueLoss = torch::mse_loss(value * 25.f, valueTargets * 25.f);
            auto loss = policyLoss + valueLoss;

            // if loss is a lot higher than trainStats.getAverageLoss() then log the state
            // and policyTargets and valueTargets if (loss.item<float>() > 2 *
            // trainStats.getAverageLoss()) {
            //     log("High loss detected, logging state and targets");
            //     log("State:\n", states);
            //     log("Policy Targets:\n", policyTargets);
            //     log("Value Targets:\n", valueTargets);
            // }

            loss.backward();

            timeit([&] { step(model); }, "step");

            trainStats.update(policyLoss.item<float>(), valueLoss.item<float>(),
                              loss.item<float>());
        }

        m_barrier.updateRequired(-1);

        return trainStats;
    }

    std::vector<std::unique_ptr<std::mutex>> m_aggregateMutexes;

    void initializeAggregateMutexes() {
        m_aggregateMutexes.clear();
        for (size_t i = 0; i < m_model->parameters().size(); ++i) {
            m_aggregateMutexes.push_back(std::make_unique<std::mutex>());
        }
    }

    void step(Network &model) {
        // barrier here to ensure all threads have finished back propagation and gradients are
        // available for aggregation
        // only do this block once (barrier returns true for only one of the threads) and not
        // for all threads

        timeit(
            [&] {
                torch::NoGradGuard no_grad; // Ensure no gradient computation happens here

                for (size_t i = 0; i < m_model->parameters().size(); ++i) {
                    // Accumulate gradients to the main model's parameters
                    auto main_param = m_model->parameters()[i];
                    auto param = model->parameters()[i];

                    auto grad = param.grad().to(m_model->device);

                    m_aggregateMutexes[i]->lock();
                    main_param.mutable_grad() += grad;
                    m_aggregateMutexes[i]->unlock();

                    // Reset the gradients for the next iteration
                    param.grad().zero_();
                }
            },
            "aggregate gradients");

        // Only continue threads once this line is reached
        if (timeit([&] { return m_barrier.barrier(); }, "barrier")) {

            // Average the gradients here (if desired, depending on your training strategy)

            timeit(
                [&] {
                    m_optimizer->step();
                    m_optimizer->zero_grad();
                },
                "step optimizer");
        }

        // Now the main model's parameters have been updated with the aggregated gradients
        // All threads can now continue and should synchronize their models with the main model
        timeit([&] { m_barrier.barrier(); }, "barrier");

        // Ensure all models are updated with the main model's parameters
        timeit([&] { synchronizeModel(model); }, "synchronizeModel");
    }

    void synchronizeModel(Network &model) {
        // Iterate over each model in m_models to update its parameters
        auto model_params = model->named_parameters();
        auto model_buffers = model->named_buffers();

        // Iterate over parameters to update them
        for (auto &param : m_model->named_parameters()) {
            auto &name = param.key();
            auto &main_param = param.value();
            auto &model_param = *model_params.find(name);

            // Ensure the parameter is copied over to the correct device
            model_param.data().copy_(main_param.data().to(model_param.device()));
        }

        // Also ensure buffers (e.g., running mean/var in BatchNorm) are synchronized
        for (auto &buffer : m_model->named_buffers()) {
            auto &name = buffer.key();
            auto &main_buffer = buffer.value();
            auto &model_buffer = *model_buffers.find(name);

            model_buffer.data().copy_(main_buffer.data().to(model_buffer.device()));
        }
    }
};
