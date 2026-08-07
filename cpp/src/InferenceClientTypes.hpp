#pragma once

#include "InferenceTypes.hpp"
#include "SearchInference.hpp"
#include "common.hpp"
#include "games/chess/ChessGameContract.hpp"

using InferenceResult = SearchInferenceResult<ChessGameContract>;

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
