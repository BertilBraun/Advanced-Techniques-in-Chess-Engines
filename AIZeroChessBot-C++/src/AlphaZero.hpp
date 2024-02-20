// This bot is heavily based on the Alpha Zero From Scratch Project by foersterrober
// (https://github.com/foersterrobert/AlphaZeroFromScratch/blob/main/9.AlphaParallel.ipynb)

#pragma once

#include "common.hpp"

#include "MoveEncoding.hpp"
#include "Network.hpp"
#include "SelfPlay.hpp"
#include "SelfPlayGame.hpp"
#include "TrainingArgs.hpp"
#include "TrainingStats.hpp"

#include "Dataset.hpp"

class AlphaZero {
public:
    AlphaZero(Network &model, torch::optim::Optimizer &optimizer, const TrainingArgs &args,
              bool doLoadLatestModel = true)
        : m_model(model), m_optimizer(optimizer), m_args(args), m_selfPlay(model, args),
          m_savePath(m_args.savePath) {

        // create save directory if it does not exist
        if (!std::filesystem::exists(m_savePath)) {
            std::filesystem::create_directories(m_savePath);
        }

        if (doLoadLatestModel) {
            loadLatestModel();
        }

        initializeCluster();
    }

    void learn() {
        LearningStats learningStats;
        for (size_t iteration = m_startingIteration; iteration < m_args.numIterations;
             ++iteration) {

            SelfPlayStats selfPlayStats;
            size_t selfPlayGamesInParallel =
                m_args.numParallelGames * m_args.numSeparateNodesOnCluster;
            size_t selfPlayIterations = m_args.numSelfPlayIterations / selfPlayGamesInParallel;

            m_model->eval(); // Set model to evaluation mode
            for (size_t i = 0; tqdm(i, selfPlayIterations, "Self-play"); ++i) {
                // Collect new memories from self-play
                selfPlayStats += timeit([&] { return m_selfPlay.selfPlay(); }, "selfPlay");
            }

            // Output stats
            std::cout << "Timeit stats:" << std::endl << get_timeit_results() << std::endl;
            std::cout << "Iteration " << (iteration + 1) << " Self Play Stats:" << std::endl
                      << selfPlayStats.toString() << std::endl;

            if (m_isRootNode) {
                TrainingStats trainStats;
                Dataset dataset(m_savePath, m_model->device, 10);

                std::cout << "Training with " << dataset.size() * m_args.batchSize << " memories\n";

                m_model->train(); // Set model to training mode
                for (size_t i = 0; tqdm(i, m_args.numEpochs, "Training"); ++i) {
                    trainStats += timeit([&] { return train(dataset); }, "train");
                }

                std::cout << "Iteration " << (iteration + 1) << " Train Stats:" << std::endl
                          << trainStats.toString() << std::endl;
                auto modelPath = saveLatestModel(iteration);
                learningStats.update(dataset.size() * m_args.batchSize, trainStats);

                // Retain 25% of the memory for the next iteration
                dataset.deleteOldMemories(m_args.retentionRate);

                // evaluateAlphaVsStockfish(modelPath);
                std::cout << "Timeit stats:" << std::endl << get_timeit_results() << std::endl;
            }

            barrier("training_done");
            loadLatestModel(); // Synchronize model across nodes/instances
        }

        std::cout << "Learning finished\n";
        std::cout << learningStats.toString() << std::endl;
    }

private:
    Network &m_model;
    torch::optim::Optimizer &m_optimizer;
    TrainingArgs m_args;
    size_t m_startingIteration = 0;
    SelfPlay m_selfPlay;
    std::filesystem::path m_savePath;
    unsigned int m_id;
    bool m_isRootNode;

    TrainingStats train(Dataset &dataset) {
        TrainingStats trainStats;

        while (dataset.hasNext()) {
            auto [states, policyTargets, valueTargets] = dataset.next();
            auto [policy, value] = m_model->forward(states);

            auto policyLoss = torch::nn::functional::cross_entropy(policy, policyTargets);
            auto valueLoss = torch::mse_loss(value, valueTargets);
            auto loss = policyLoss + valueLoss;

            m_optimizer.zero_grad();
            loss.backward();
            m_optimizer.step();

            trainStats.update(policyLoss.item<float>(), valueLoss.item<float>(),
                              loss.item<float>());
        }

        return trainStats;
    }

    void loadLatestModel() {
        std::filesystem::path lastTrainingConfigPath = m_savePath / CONFIG_FILE_NAME;

        if (std::filesystem::exists(lastTrainingConfigPath)) {
            try {
                size_t loadedIteration = 0;
                std::string modelPath, optimizerPath;
                // Load the training configuration
                if (!loadConfiguration(lastTrainingConfigPath, modelPath, optimizerPath,
                                       loadedIteration)) {
                    std::cerr << "Failed to load training configuration" << std::endl;
                    return;
                }

                // Load model state
                if (std::filesystem::exists(modelPath)) {
                    torch::load(m_model, modelPath);
                } else {
                    std::cerr << "Saved model file not found: " << modelPath << std::endl;
                    return;
                }

                // Load optimizer state
                if (std::filesystem::exists(optimizerPath)) {
                    torch::load(m_optimizer, optimizerPath);
                } else {
                    std::cerr << "Saved optimizer file not found: " << optimizerPath << std::endl;
                    return;
                }

                // Assuming you want to continue from the next iteration
                m_startingIteration = loadedIteration + 1;
                std::cout << "Model and optimizer loaded from iteration " << loadedIteration
                          << std::endl;
            } catch (const torch::Error &e) {
                std::cerr << "Error loading model and optimizer states: " << e.what() << std::endl;
            }
        } else {
            std::cout << "No model and optimizer found, starting from scratch" << std::endl;
        }
    }

    std::filesystem::path saveLatestModel(int iteration) const {
        std::filesystem::path modelPath =
            m_savePath / ("model_" + std::to_string(iteration) + ".pt");
        std::filesystem::path optimizerPath =
            m_savePath / ("optimizer_" + std::to_string(iteration) + ".pt");
        std::filesystem::path lastTrainingConfigPath = m_savePath / CONFIG_FILE_NAME;

        // Save model state
        torch::save(m_model, modelPath.string());

        // Save optimizer state
        torch::save(m_optimizer, optimizerPath.string());

        // Save training configuration
        saveConfiguration(lastTrainingConfigPath, modelPath.string(), optimizerPath.string(),
                          iteration);

        std::cout << "Model and optimizer saved at iteration " << iteration << std::endl;

        return modelPath;
    }

    void saveConfiguration(const std::filesystem::path &path, const std::string &modelPath,
                           const std::string &optimizerPath, int iteration) const {
        std::ofstream configFile(path);
        if (!configFile.is_open()) {
            std::cerr << "Failed to open config file for writing: " << path << std::endl;
            return;
        }

        configFile << "model=" << modelPath << "\n";
        configFile << "optimizer=" << optimizerPath << "\n";
        configFile << "iteration=" << iteration << "\n";
        configFile.close();

        std::cout << "Configuration saved to " << path << std::endl;
    }

    bool loadConfiguration(const std::filesystem::path &path, std::string &modelPath,
                           std::string &optimizerPath, size_t &iteration) const {
        std::ifstream configFile(path);
        if (!configFile.is_open()) {
            std::cerr << "Failed to open config file for reading: " << path << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(configFile, line)) {
            auto delimiterPos = line.find('=');
            if (delimiterPos != std::string::npos) {
                std::string key = line.substr(0, delimiterPos);
                std::string value = line.substr(delimiterPos + 1);

                if (key == "model") {
                    modelPath = value;
                } else if (key == "optimizer") {
                    optimizerPath = value;
                } else if (key == "iteration") {
                    iteration = std::stoi(value);
                }
            }
        }

        configFile.close();
        return true;
    }

    void initializeCluster() {
        m_id = rand();

        std::filesystem::create_directories(COMMUNICATION_DIR);
        // Delete everything in the communication directory
        for (const auto &entry : std::filesystem::directory_iterator(COMMUNICATION_DIR)) {
            std::filesystem::remove(entry.path());
        }

        std::cout << "Node " << m_id << " initialized\n";

        auto logFile = COMMUNICATION_DIR / (std::to_string(m_id) + ".txt");

        // Wait for all nodes to initialize
        while (true) {
            std::ofstream(logFile).close(); // Create an empty file to signal initialization

            size_t initializedNodes =
                std::distance(std::filesystem::directory_iterator(COMMUNICATION_DIR),
                              std::filesystem::directory_iterator{});
            if (initializedNodes == m_args.numSeparateNodesOnCluster) {
                break;
            }

            std::cout << "Waiting for " << m_args.numSeparateNodesOnCluster - initializedNodes
                      << " nodes to initialize\n";
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        std::cout << "All nodes initialized\n";

        // Determine if this node is the root node
        unsigned int minId = m_id;
        for (const auto &entry : std::filesystem::directory_iterator(COMMUNICATION_DIR)) {
            unsigned int id = std::stoul(entry.path().stem());
            minId = std::min(minId, id);
        }
        m_isRootNode = (m_id == minId);
        if (m_isRootNode) {
            std::cout << "I am the root node\n";
        } else {
            std::cout << "I am not the root node\n";
        }

        std::this_thread::sleep_for(std::chrono::seconds(5));

        // Remove the initialization file
        std::filesystem::remove(logFile);
    }

    void barrier(const std::string &name) const {
        auto logFile =
            COMMUNICATION_DIR / (std::to_string(m_id) + (name.empty() ? "" : "_" + name) + ".txt");

        std::cout << "Node " << m_id << " reached the barrier " << name << "\n";
        std::ofstream(logFile).close(); // Signal this node has reached the barrier

        // Wait for all nodes to reach the barrier
        while (true) {
            // Count the number of files in the communication directory that end with the name
            size_t writtenFiles = 0;
            for (const auto &entry : std::filesystem::directory_iterator(COMMUNICATION_DIR)) {
                if (entry.path().stem().string().find(name) != std::string::npos) {
                    ++writtenFiles;
                }
            }

            std::this_thread::sleep_for(std::chrono::seconds(5));

            if (writtenFiles == m_args.numSeparateNodesOnCluster) {
                break;
            }
        }

        std::cout << "All nodes have reached the barrier " << name << "\n";

        // Remove the barrier signal file
        std::filesystem::remove(logFile);

        std::this_thread::sleep_for(std::chrono::seconds(20));
    }
};
