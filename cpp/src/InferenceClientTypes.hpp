#pragma once

#include "InferenceTypes.hpp"
#include "common.hpp"

enum class WdlIndex : size_t { Win = 0, Draw = 1, Loss = 2, Count = 3 };
constexpr size_t WDL_OUTPUT_SIZE = static_cast<size_t>(WdlIndex::Count);

struct WdlPrediction {
    float win;
    float draw;
    float loss;

    [[nodiscard]] float expectedValue() const { return win - loss; }
    [[nodiscard]] float value() const { return expectedValue(); }
    bool operator==(const WdlPrediction &) const = default;
};

using OutcomeProbabilities = WdlPrediction;

struct InferenceResult {
    std::vector<MoveScore> moves;
    WdlPrediction outcome;

    [[nodiscard]] float value() const { return outcome.expectedValue(); }
};

struct InferenceClientParams {
    int device_id;
    std::string currentModelPath;
    int maxBatchSize;
    int microsecondsTimeoutInferenceThread;
    InferenceDevice device;

    InferenceClientParams(int device_id, std::string currentModelPath, int maxBatchSize,
                          int microsecondsTimeoutInferenceThread,
                          InferenceDevice device = InferenceDevice::Auto)
        : device_id(device_id), currentModelPath(std::move(currentModelPath)),
          maxBatchSize(maxBatchSize),
          microsecondsTimeoutInferenceThread(microsecondsTimeoutInferenceThread), device(device) {
        if (maxBatchSize <= 0) {
            throw std::invalid_argument("maxBatchSize must be positive");
        }
        if (microsecondsTimeoutInferenceThread < 0) {
            throw std::invalid_argument("microsecondsTimeoutInferenceThread cannot be negative");
        }
    }
};
