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

class AlphaZero {
public:
    AlphaZero(Network &model, torch::optim::Optimizer &optimizer, const TrainingArgs &args,
              bool doLoadLatestModel = true)
        : m_model(model), m_optimizer(optimizer), args(args), m_selfPlay(model, args),
          m_savePath(args.savePath) {

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
        for (int iteration = m_startingIteration; iteration < args.numIterations; ++iteration) {
            TrainingStats trainStats;

            int selfPlayGamesInParallel = args.numParallelGames * args.numSeparateNodesOnCluster;
            int selfPlayIterations = args.numSelfPlayIterations / selfPlayGamesInParallel;

            m_model->eval(); // Set model to evaluation mode
            for (int i = 0; tqdm(i, selfPlayIterations, "Self-play"); ++i) {
                // Collect new memories from self-play
                timeit([&] { m_selfPlay.selfPlay(); }, "selfPlay");
            }

            // Output timeit stats
            std::cout << "Timeit stats:" << std::endl
                      << get_timeit_results_and_reset() << std::endl;

            if (m_isRootNode) {
                auto totalDatasetSize = getTotalDatasetSize();
                std::cout << "Training with " << totalDatasetSize << " memories\n";

                m_model->train(); // Set model to training mode
                for (int i = 0; tqdm(i, args.numEpochs, "Training"); ++i) {
                    trainStats += train(); // Accumulate training stats
                }

                std::cout << "Iteration " << (iteration + 1) << ": " << trainStats.toString()
                          << std::endl;
                auto modelPath = saveLatestModel(iteration);
                learningStats.update(totalDatasetSize, trainStats);

                // Retain 25% of the memory for the next iteration
                deleteOldMemories();

                // evaluateAlphaVsStockfish(modelPath);
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
    TrainingArgs args;
    unsigned int m_startingIteration = 0;
    SelfPlay m_selfPlay;
    std::filesystem::path m_savePath;
    unsigned int m_id;
    bool m_isRootNode;

    TrainingStats train() {
        auto memoryPaths = getMemoryPaths();
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(memoryPaths.begin(), memoryPaths.end(), g);

        TrainingStats trainStats;

        int batchSize = args.batchSize;
        for (auto memoryPath : memoryPaths) {
            auto [states, policyTargets, valueTargets] = loadMemory(memoryPath);

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
                int loadedIteration;
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
                           std::string &optimizerPath, int &iteration) const {
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

    int getTotalDatasetSize() const {
        int totalDatasetSize = 0;
        for (const auto &entry :
             std::filesystem::directory_iterator(m_savePath / MEMORY_DIR_NAME)) {
            if (entry.is_directory()) {
                // if all files are present, increment the totalDatasetSize
                if (std::filesystem::exists(entry.path() / "states.pt") &&
                    std::filesystem::exists(entry.path() / "policyTargets.pt") &&
                    std::filesystem::exists(entry.path() / "valueTargets.pt")) {
                    totalDatasetSize++;
                }
            }
        }
        return totalDatasetSize;
    }

    void deleteOldMemories(int retentionFactor = 25) const {
        for (const auto &entry :
             std::filesystem::directory_iterator(m_savePath / MEMORY_DIR_NAME)) {
            if (entry.is_directory()) {
                if (std::filesystem::exists(entry.path() / "states.pt") &&
                    std::filesystem::exists(entry.path() / "policyTargets.pt") &&
                    std::filesystem::exists(entry.path() / "valueTargets.pt")) {
                    if (rand() % 100 > retentionFactor) {
                        // delete the entries in the folder as well as the folder itself
                        std::filesystem::remove_all(entry.path());
                    }
                }
            }
        }
    }

    std::vector<std::filesystem::path> getMemoryPaths() const {
        std::vector<std::filesystem::path> memoryPaths;
        for (const auto &entry :
             std::filesystem::directory_iterator(m_savePath / MEMORY_DIR_NAME)) {
            if (entry.is_directory()) {
                if (std::filesystem::exists(entry.path() / "states.pt") &&
                    std::filesystem::exists(entry.path() / "policyTargets.pt") &&
                    std::filesystem::exists(entry.path() / "valueTargets.pt")) {
                    memoryPaths.push_back(entry.path());
                }
            }
        }
        return memoryPaths;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    loadMemory(const std::filesystem::path &memoryPath) const {
        torch::Tensor states, policyTargets, valueTargets;

        torch::load(states, (memoryPath / "states.pt").string(), m_model->device);
        torch::load(policyTargets, (memoryPath / "policyTargets.pt").string(), m_model->device);
        torch::load(valueTargets, (memoryPath / "valueTargets.pt").string(), m_model->device);

        return std::make_tuple(states, policyTargets, valueTargets);
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
            if (initializedNodes == args.numSeparateNodesOnCluster) {
                break;
            }

            std::cout << "Waiting for " << args.numSeparateNodesOnCluster - initializedNodes
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

            if (writtenFiles == args.numSeparateNodesOnCluster) {
                break;
            }
        }

        std::cout << "All nodes have reached the barrier " << name << "\n";

        // Remove the barrier signal file
        std::filesystem::remove(logFile);

        std::this_thread::sleep_for(std::chrono::seconds(20));
    }
};
