#pragma once

#include "common.hpp"

struct OutcomeProbabilities {
    float win;
    float draw;
    float loss;

    [[nodiscard]] float value() const { return win - loss; }
    bool operator==(const OutcomeProbabilities &) const = default;
};

struct InferenceResult {
    std::vector<MoveScore> moves;
    OutcomeProbabilities outcome;
};

struct InferenceStatistics {
    float cacheHitRate = 0.0f;
    size_t evaluations = 0;
    size_t cacheHits = 0;
    size_t uniquePositions = 0;
    size_t cacheSizeMB = 0;
    size_t cacheCapacity = 0;
    size_t cacheEvictions = 0;
    size_t cacheFingerprintCollisions = 0;
    std::vector<float> nnOutputValueDistribution;
    size_t modelInferenceCalls = 0;
    size_t modelInferencePositions = 0;
    std::vector<size_t> modelBatchSizeHistogram;
    float averageNumberOfPositionsInInferenceCall = 0.0f;
};

struct InferenceClientParams {
    int device_id;
    std::string currentModelPath;
    int maxBatchSize;
    int microsecondsTimeoutInferenceThread;
    size_t cacheCapacity;

    InferenceClientParams(int device_id, std::string currentModelPath, int maxBatchSize,
                          int microsecondsTimeoutInferenceThread, size_t cacheCapacity)
        : device_id(device_id), currentModelPath(std::move(currentModelPath)),
          maxBatchSize(maxBatchSize),
          microsecondsTimeoutInferenceThread(microsecondsTimeoutInferenceThread),
          cacheCapacity(cacheCapacity) {}
};
