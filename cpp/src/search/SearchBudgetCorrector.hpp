#pragma once

#include "search/InferenceTypes.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>

struct SearchBudgetSelectionFeatures {
    double top_visit_share;
    double policy_entropy;
    double ply;
    double baseline_visits;
    double source_generation;
};

// TorchScript curve corrector published by the Python calibration loop. Input is the predicted
// curve followed by top visit share, policy entropy, ply, baseline visits and source generation
// (a binding contract with src/search_budget/corrector.py); output is one additive log-KL
// correction per grid point. Standardisation is folded into the module, so evaluation is raw.
class SearchBudgetCurveCorrector {
public:
    static constexpr std::size_t FEATURE_COUNT = SEARCH_BUDGET_CURVE_POINTS + 5;

    // Loads on CPU and validates the module with a probe forward, mirroring inference refresh.
    explicit SearchBudgetCurveCorrector(const std::string &modelPath);
    ~SearchBudgetCurveCorrector();
    SearchBudgetCurveCorrector(const SearchBudgetCurveCorrector &) = delete;
    SearchBudgetCurveCorrector &operator=(const SearchBudgetCurveCorrector &) = delete;

    [[nodiscard]] std::array<double, SEARCH_BUDGET_CURVE_POINTS>
    correction(const SearchBudgetCurvePrediction &prediction,
               const SearchBudgetSelectionFeatures &features) const;

private:
    struct Implementation;
    std::unique_ptr<Implementation> m_implementation;
};
