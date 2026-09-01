#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>

// The stop-predictor input contract (adaptive-stopping plan section 4.2). Order is a binding
// contract with src/search_stopping/features.py::STOP_PREDICTOR_FEATURE_NAMES.
inline constexpr std::size_t STOP_PREDICTOR_FEATURE_COUNT = 17;
using StopPredictorFeatures = std::array<double, STOP_PREDICTOR_FEATURE_COUNT>;

// TorchScript stop predictor published by the Python calibration loop. Output is the probability
// that the search is still uncertain (keep searching). Standardisation is folded into the module,
// so evaluation is raw; loads on CPU and validates the module with a probe forward.
class SearchStopPredictor {
public:
    explicit SearchStopPredictor(const std::string &modelPath);
    ~SearchStopPredictor();
    SearchStopPredictor(const SearchStopPredictor &) = delete;
    SearchStopPredictor &operator=(const SearchStopPredictor &) = delete;

    [[nodiscard]] double uncertainty(const StopPredictorFeatures &features) const;

private:
    struct Implementation;
    std::unique_ptr<Implementation> m_implementation;
};
