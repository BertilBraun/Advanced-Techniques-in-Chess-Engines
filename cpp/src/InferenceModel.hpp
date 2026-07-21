#pragma once

#include "common.hpp"

namespace inference_model_detail {
template <typename NamedTensorList>
void validateNamedTensors(const NamedTensorList &currentTensors,
                          const NamedTensorList &updatedTensors,
                          const std::string &tensorKind) {
    if (currentTensors.size() != updatedTensors.size()) {
        throw std::invalid_argument("Updated inference model has a different number of " +
                                    tensorKind + "s");
    }

    auto current = currentTensors.begin();
    auto updated = updatedTensors.begin();
    while (current != currentTensors.end()) {
        const auto currentTensor = *current;
        const auto updatedTensor = *updated;
        if (currentTensor.name != updatedTensor.name) {
            throw std::invalid_argument("Updated inference model " + tensorKind +
                                        " names do not match");
        }
        if (currentTensor.value.sizes() != updatedTensor.value.sizes()) {
            throw std::invalid_argument("Updated inference model " + tensorKind + " " +
                                        currentTensor.name + " has a different shape");
        }
        ++current;
        ++updated;
    }
}

template <typename NamedTensorList>
void copyNamedTensors(const NamedTensorList &currentTensors,
                      const NamedTensorList &updatedTensors) {
    auto current = currentTensors.begin();
    auto updated = updatedTensors.begin();
    while (current != currentTensors.end()) {
        const auto currentTensor = *current;
        const auto updatedTensor = *updated;
        currentTensor.value.copy_(updatedTensor.value);
        ++current;
        ++updated;
    }
}
} // namespace inference_model_detail

[[nodiscard]] inline torch::jit::script::Module
loadInferenceModel(const std::string &modelPath, const torch::Device &device,
                   const torch::Dtype dataType) {
    std::string modelPathToLoad = modelPath;
    if (!modelPathToLoad.ends_with(".jit.pt") && !modelPathToLoad.ends_with(".pt")) {
        throw std::invalid_argument("Model path must end with '.jit.pt' or '.pt'");
    }
    if (!modelPathToLoad.ends_with(".jit.pt")) {
        modelPathToLoad = modelPathToLoad.substr(0, modelPathToLoad.size() - 3) + ".jit.pt";
    }
    if (!std::filesystem::exists(modelPathToLoad)) {
        throw std::invalid_argument("Model file does not exist: " + modelPathToLoad);
    }

    torch::jit::script::Module model = torch::jit::load(modelPathToLoad, device);
    model.to(dataType);
    model.eval();
    return model;
}

inline void updateInferenceModel(torch::jit::script::Module &model,
                                 const std::string &modelPath, const torch::Device &device,
                                 const torch::Dtype dataType) {
    torch::jit::script::Module updatedModel = loadInferenceModel(modelPath, device, dataType);
    torch::NoGradGuard noGrad;
    const auto currentParameters = model.named_parameters();
    const auto updatedParameters = updatedModel.named_parameters();
    const auto currentBuffers = model.named_buffers();
    const auto updatedBuffers = updatedModel.named_buffers();
    inference_model_detail::validateNamedTensors(currentParameters, updatedParameters,
                                                 "parameter");
    inference_model_detail::validateNamedTensors(currentBuffers, updatedBuffers, "buffer");
    inference_model_detail::copyNamedTensors(currentParameters, updatedParameters);
    inference_model_detail::copyNamedTensors(currentBuffers, updatedBuffers);
}
