#pragma once

#include "common.hpp"

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

enum class InferenceDevice { Auto, Cpu, Cuda };

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
    std::uint64_t treeSelectionNanoseconds = 0;
    std::uint64_t boardEncodingNanoseconds = 0;
    std::uint64_t resultProcessingNanoseconds = 0;
    std::uint64_t treeBackupNanoseconds = 0;
    std::uint64_t treeOwnerWaitNanoseconds = 0;
    std::uint64_t directInferenceNanoseconds = 0;
    float directWorkerUtilization = 0.0F;
};

struct InferenceClientParams {
    int device_id;
    std::string currentModelPath;
    int maxBatchSize;
    int microsecondsTimeoutInferenceThread;
    size_t cacheCapacity;
    InferenceDevice device;

    InferenceClientParams(int device_id, std::string currentModelPath, int maxBatchSize,
                          int microsecondsTimeoutInferenceThread, size_t cacheCapacity,
                          InferenceDevice device = InferenceDevice::Auto)
        : device_id(device_id), currentModelPath(std::move(currentModelPath)),
          maxBatchSize(maxBatchSize),
          microsecondsTimeoutInferenceThread(microsecondsTimeoutInferenceThread),
          cacheCapacity(cacheCapacity), device(device) {
        if (maxBatchSize <= 0) {
            throw std::invalid_argument("maxBatchSize must be positive");
        }
        if (microsecondsTimeoutInferenceThread < 0) {
            throw std::invalid_argument("microsecondsTimeoutInferenceThread cannot be negative");
        }
    }
};
