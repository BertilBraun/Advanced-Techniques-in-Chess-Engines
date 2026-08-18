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
    SdpaBackend sdpa_backend;

    InferenceConfiguration(int deviceId, std::string modelPath,
                           InferenceDevice inferenceDevice = InferenceDevice::Auto,
                           SdpaBackend sdpaBackend = SdpaBackend::Automatic)
        : device_id(deviceId), model_path(std::move(modelPath)), device(inferenceDevice),
          sdpa_backend(sdpaBackend) {
        if (model_path.empty()) {
            throw std::invalid_argument("Inference model path must not be empty");
        }
    }
};
