#pragma once

#include "common.hpp"

#include "Network.hpp"
#include "TrainingArgs.hpp"

class AlphaZeroBase {
public:
    AlphaZeroBase(Network &model, const TrainingArgs &args,
                  torch::optim::Optimizer *optimizer = nullptr)
        : m_model(model), m_args(args), m_savePath(m_args.savePath), m_optimizer(optimizer) {

        // create save directory if it does not exist
        if (!std::filesystem::exists(m_savePath)) {
            std::filesystem::create_directories(m_savePath);
        }

        loadLatestModel();
    }

    Network &m_model;
    TrainingArgs m_args;
    std::filesystem::path m_savePath;
    torch::optim::Optimizer *m_optimizer;
    size_t m_startingIteration = 0;

    void loadLatestModel() {
        std::filesystem::path lastTrainingConfigPath = m_savePath / CONFIG_FILE_NAME;

        if (std::filesystem::exists(lastTrainingConfigPath)) {
            size_t loadedIteration = 0;
            std::string modelPath, optimizerPath;
            // Load the training configuration
            if (!loadConfiguration(lastTrainingConfigPath, modelPath, optimizerPath,
                                   loadedIteration)) {
                log("Failed to load training configuration");
                return;
            }

            // Load model state
            try {
                torch::load(m_model, modelPath);
                m_model->device = torch::cuda::is_available() && torch::cuda::device_count() > 0
                                      ? torch::kCUDA
                                      : torch::kCPU;
                m_model->to(m_model->device);
            } catch (const torch::Error &e) {
                log("Error loading model state:", modelPath);
                log("Error:", e.what());
                return;
            }

            // Load optimizer state
            try {
                if (m_optimizer != nullptr)
                    torch::load(*m_optimizer, optimizerPath);
            } catch (const torch::Error &e) {
                log("Error loading optimizer state:", optimizerPath);
                log("Error:", e.what());
                return;
            }

            // Assuming you want to continue from the next iteration
            m_startingIteration = loadedIteration + 1;
            log("Model and optimizer loaded from iteration", loadedIteration);
        } else {
            log("No model and optimizer found, starting from scratch");
        }
    }

    void saveLatestModel(size_t iteration) const {
        std::filesystem::path modelPath =
            m_savePath / ("model_" + std::to_string(iteration) + ".pt");
        std::filesystem::path optimizerPath =
            m_savePath / ("optimizer_" + std::to_string(iteration) + ".pt");
        std::filesystem::path lastTrainingConfigPath = m_savePath / CONFIG_FILE_NAME;

        // Save model state
        m_model->to(torch::kCPU);
        torch::save(m_model, modelPath.string());
        m_model->to(m_model->device);

        // Save optimizer state
        if (m_optimizer != nullptr)
            torch::save(*m_optimizer, optimizerPath.string());

        // Save training configuration
        saveConfiguration(lastTrainingConfigPath, modelPath, optimizerPath, iteration);

        log("Model and optimizer saved at iteration", iteration);
    }

private:
    void saveConfiguration(const std::filesystem::path &path,
                           const std::filesystem::path &modelPath,
                           const std::filesystem::path &optimizerPath, size_t iteration) const {
        std::ofstream configFile(path);
        if (!configFile.is_open()) {
            log("Failed to open config file for writing:", path);
            return;
        }

        // Save the absolute paths to the model and optimizer files

        auto absModelPath = std::filesystem::absolute(modelPath);
        auto absOptimizerPath = std::filesystem::absolute(optimizerPath);

        configFile << "model=" << absModelPath << "\n";
        configFile << "optimizer=" << absOptimizerPath << "\n";
        configFile << "iteration=" << iteration << "\n";
        configFile.close();

        log("Configuration saved to", path);
    }

    bool loadConfiguration(const std::filesystem::path &path, std::string &modelPath,
                           std::string &optimizerPath, size_t &iteration) const {
        std::ifstream configFile(path);
        if (!configFile.is_open()) {
            log("Failed to open config file for reading:", path);
            return false;
        }

        std::string line;
        while (std::getline(configFile, line)) {
            auto parts = split(line, '=');
            if (parts.size() == 2) {
                std::string key = parts[0];
                std::string value = parts[1];

                if (key == "model") {
                    modelPath = value;
                } else if (key == "optimizer") {
                    optimizerPath = value;
                } else if (key == "iteration") {
                    iteration = std::stoi(value);
                }
            }
        }

        // NOTE: The paths are saved as absolute paths, so we need to convert them to relative paths
        //       because libtorch's load function apparently doesn't like absolute paths
        // NOTE: Somehow the std::filesystem::relative function doesn't work as expected, so we'll
        //       just strip the common prefix from the paths

        // make the model and optimizer paths relative to the cwd
        std::string cwd = std::filesystem::current_path().string();
        modelPath = modelPath.substr(cwd.size() + 2);
        // strip trailing '"' if it exists
        if (modelPath.back() == '"') {
            modelPath.pop_back();
        }
        optimizerPath = optimizerPath.substr(cwd.size() + 2);
        // strip trailing '"' if it exists
        if (optimizerPath.back() == '"') {
            optimizerPath.pop_back();
        }

        configFile.close();
        return true;
    }
};
