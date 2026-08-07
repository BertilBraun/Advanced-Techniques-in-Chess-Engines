#pragma once

#include "search/InferenceTypes.hpp"

#include <stdexcept>
#include <string>

// Carries the model, device, and batching configuration for one inference pipeline.
#include <utility>

struct InferenceConfiguration {
    int device_id;
    std::string model_path;
    InferenceDevice device;

    InferenceConfiguration(int deviceId, std::string modelPath,
                           InferenceDevice inferenceDevice = InferenceDevice::Auto)
        : device_id(deviceId), model_path(std::move(modelPath)), device(inferenceDevice) {
        if (model_path.empty()) {
            throw std::invalid_argument("Inference model path must not be empty");
        }
    }
};
