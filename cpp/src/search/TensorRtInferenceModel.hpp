#pragma once

#include "common.hpp"
#include "search/InferenceTypes.hpp"

#include <memory>
#include <string>

struct InferenceOutput;

class TensorRtInferenceModel {
public:
    TensorRtInferenceModel(const std::string &enginePath, int deviceId, size_t maximumBatchSize,
                           InferenceDimensions dimensions);
    ~TensorRtInferenceModel();

    TensorRtInferenceModel(const TensorRtInferenceModel &) = delete;
    TensorRtInferenceModel &operator=(const TensorRtInferenceModel &) = delete;

    void forward(const torch::Tensor &encodedInput, size_t batchSize, InferenceOutput &output);

private:
    class Implementation;
    std::unique_ptr<Implementation> m_implementation;
};
