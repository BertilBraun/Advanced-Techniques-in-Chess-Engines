#pragma once

#include "search/InferenceTypes.hpp"

#include <stdexcept>
#include <string>
#include <utility>

struct InferenceRuntimeParameters {
    int device_id;
    std::string model_path;
    InferenceDevice device;

    InferenceRuntimeParameters(int deviceId, std::string modelPath,
                               InferenceDevice inferenceDevice = InferenceDevice::Auto)
        : device_id(deviceId), model_path(std::move(modelPath)), device(inferenceDevice) {
        if (model_path.empty()) {
            throw std::invalid_argument("Inference model path must not be empty");
        }
    }
};
