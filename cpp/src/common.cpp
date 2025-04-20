#include "common.hpp"

thread_local std::mt19937 _random_engine(std::random_device{}());

std::vector<float> dirichlet(float alpha, size_t n) {
    // Sample from a Dirichlet distribution with parameter alpha.
    std::gamma_distribution<float> gamma(alpha, 1.0);

    std::vector<float> noise(n);
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        noise[i] = gamma(_random_engine);
        sum += noise[i];
    }

    float norm = 1.0f / sum;
    for (size_t i = 0; i < n; i++) {
        noise[i] *= norm;
    }

    return noise;
}

std::pair<std::string, int> get_latest_iteration_save_path(const std::string &savePath) {
    // Models are saved in the savePath folder numbered starting from 1 up to the latest iteration.
    // For example: "{savePath}/model_1.pt", "{savePath}/model_2.pt", etc.
    // The function returns the latest model file path and its iteration number.

    for (int i : range(500, 0, -1)) {
        const std::string modelPath = savePath + "/model_" + std::to_string(i) + ".jit.pt";
        if (std::filesystem::exists(std::filesystem::path(modelPath))) {
            return {modelPath, i};
        }
    }
    throw std::runtime_error("No model found in the save path.");
}
size_t current_time_step() {
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}