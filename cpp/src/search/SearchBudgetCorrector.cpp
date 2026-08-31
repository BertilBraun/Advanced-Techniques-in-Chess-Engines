#include "search/SearchBudgetCorrector.hpp"

#include <torch/script.h>

#include <cmath>
#include <filesystem>
#include <mutex>
#include <stdexcept>

struct SearchBudgetCurveCorrector::Implementation {
    mutable std::mutex forward_mutex;
    mutable torch::jit::script::Module module;

    [[nodiscard]] torch::Tensor forward(const torch::Tensor &input) const {
        torch::InferenceMode inferenceMode;
        const std::lock_guard<std::mutex> guard(forward_mutex);
        return module.forward({input}).toTensor();
    }
};

SearchBudgetCurveCorrector::SearchBudgetCurveCorrector(const std::string &modelPath)
    : m_implementation(std::make_unique<Implementation>()) {
    if (!std::filesystem::exists(modelPath)) {
        throw std::invalid_argument("Search-budget corrector file does not exist: " + modelPath);
    }
    m_implementation->module = torch::jit::load(modelPath, torch::Device(torch::kCPU));
    m_implementation->module.eval();
    const torch::Tensor probe = m_implementation->forward(
        torch::zeros({1, static_cast<std::int64_t>(FEATURE_COUNT)}, torch::kFloat32));
    if (probe.dim() != 2 || probe.size(0) != 1 ||
        probe.size(1) != static_cast<std::int64_t>(SEARCH_BUDGET_CURVE_POINTS) ||
        !torch::isfinite(probe).all().item<bool>()) {
        throw std::invalid_argument(
            "Search-budget corrector must map the feature vector to one finite correction per "
            "grid point: " +
            modelPath);
    }
}

SearchBudgetCurveCorrector::~SearchBudgetCurveCorrector() = default;

std::array<double, SEARCH_BUDGET_CURVE_POINTS>
SearchBudgetCurveCorrector::correction(const SearchBudgetCurvePrediction &prediction,
                                       const SearchBudgetSelectionFeatures &features) const {
    torch::Tensor input = torch::empty({1, static_cast<std::int64_t>(FEATURE_COUNT)},
                                       torch::TensorOptions().dtype(torch::kFloat32));
    float *values = input.data_ptr<float>();
    for (std::size_t index = 0; index < SEARCH_BUDGET_CURVE_POINTS; ++index) {
        values[index] = prediction[index];
    }
    values[SEARCH_BUDGET_CURVE_POINTS + 0] = static_cast<float>(features.top_visit_share);
    values[SEARCH_BUDGET_CURVE_POINTS + 1] = static_cast<float>(features.policy_entropy);
    values[SEARCH_BUDGET_CURVE_POINTS + 2] = static_cast<float>(features.ply);
    values[SEARCH_BUDGET_CURVE_POINTS + 3] = static_cast<float>(features.baseline_visits);
    values[SEARCH_BUDGET_CURVE_POINTS + 4] = static_cast<float>(features.source_generation);
    const torch::Tensor output = m_implementation->forward(input).to(torch::kFloat32).contiguous();
    if (output.dim() != 2 || output.size(0) != 1 ||
        output.size(1) != static_cast<std::int64_t>(SEARCH_BUDGET_CURVE_POINTS)) {
        throw std::runtime_error("Search-budget corrector returned the wrong output shape");
    }
    std::array<double, SEARCH_BUDGET_CURVE_POINTS> corrections{};
    const float *outputs = output.data_ptr<float>();
    for (std::size_t index = 0; index < SEARCH_BUDGET_CURVE_POINTS; ++index) {
        corrections[index] = static_cast<double>(outputs[index]);
        if (!std::isfinite(corrections[index])) {
            throw std::runtime_error("Search-budget corrector produced a non-finite correction");
        }
    }
    return corrections;
}
