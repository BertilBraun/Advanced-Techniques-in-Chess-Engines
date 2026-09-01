#include "search/SearchStopPredictor.hpp"

#include <torch/script.h>

#include <cmath>
#include <filesystem>
#include <mutex>
#include <stdexcept>

struct SearchStopPredictor::Implementation {
    mutable std::mutex forward_mutex;
    mutable torch::jit::script::Module module;

    [[nodiscard]] torch::Tensor forward(const torch::Tensor &input) const {
        torch::InferenceMode inferenceMode;
        const std::lock_guard<std::mutex> guard(forward_mutex);
        return module.forward({input}).toTensor();
    }
};

SearchStopPredictor::SearchStopPredictor(const std::string &modelPath)
    : m_implementation(std::make_unique<Implementation>()) {
    if (!std::filesystem::exists(modelPath)) {
        throw std::invalid_argument("Stop-predictor file does not exist: " + modelPath);
    }
    m_implementation->module = torch::jit::load(modelPath, torch::Device(torch::kCPU));
    m_implementation->module.eval();
    const torch::Tensor probe = m_implementation->forward(torch::zeros(
        {1, static_cast<std::int64_t>(STOP_PREDICTOR_FEATURE_COUNT)}, torch::kFloat32));
    const bool validShape = probe.dim() == 2 && probe.size(0) == 1 && probe.size(1) == 1;
    if (!validShape || !torch::isfinite(probe).all().item<bool>() ||
        probe.lt(0.0).any().item<bool>() || probe.gt(1.0).any().item<bool>()) {
        throw std::invalid_argument(
            "Stop predictor must map the feature vector to one probability: " + modelPath);
    }
}

SearchStopPredictor::~SearchStopPredictor() = default;

double SearchStopPredictor::uncertainty(const StopPredictorFeatures &features) const {
    torch::Tensor input = torch::empty({1, static_cast<std::int64_t>(STOP_PREDICTOR_FEATURE_COUNT)},
                                       torch::TensorOptions().dtype(torch::kFloat32));
    float *values = input.data_ptr<float>();
    for (std::size_t index = 0; index < STOP_PREDICTOR_FEATURE_COUNT; ++index) {
        values[index] = static_cast<float>(features[index]);
    }
    const torch::Tensor output = m_implementation->forward(input).to(torch::kFloat32).contiguous();
    if (output.dim() != 2 || output.size(0) != 1 || output.size(1) != 1) {
        throw std::runtime_error("Stop predictor returned the wrong output shape");
    }
    const double probability = static_cast<double>(output.data_ptr<float>()[0]);
    if (!std::isfinite(probability) || probability < 0.0 || probability > 1.0) {
        throw std::runtime_error("Stop predictor produced an invalid probability");
    }
    return probability;
}
