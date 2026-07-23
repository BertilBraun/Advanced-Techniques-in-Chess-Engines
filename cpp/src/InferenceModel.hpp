#pragma once

#include "BoardEncoding.hpp"
#include "InferenceClientTypes.hpp"
#include "common.hpp"

namespace inference_model_detail {
template <typename NamedTensorList>
void validateNamedTensors(const NamedTensorList &currentTensors,
                          const NamedTensorList &updatedTensors, const std::string &tensorKind) {
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

[[nodiscard]] inline torch::jit::script::Module loadInferenceModel(const std::string &modelPath,
                                                                   const torch::Device &device,
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

using PreparedInferenceModel = std::unique_ptr<torch::jit::script::Module>;

[[nodiscard]] inline PreparedInferenceModel
prepareInferenceModelUpdate(const torch::jit::script::Module &model, const std::string &modelPath,
                            const torch::Device &device, const torch::Dtype dataType,
                            const torch::Tensor &validationInput) {
    auto updatedModel = std::make_unique<torch::jit::script::Module>(
        loadInferenceModel(modelPath, device, dataType));
    const auto currentParameters = model.named_parameters();
    const auto updatedParameters = updatedModel->named_parameters();
    const auto currentBuffers = model.named_buffers();
    const auto updatedBuffers = updatedModel->named_buffers();
    inference_model_detail::validateNamedTensors(currentParameters, updatedParameters, "parameter");
    inference_model_detail::validateNamedTensors(currentBuffers, updatedBuffers, "buffer");
    torch::InferenceMode inferenceMode;
    const torch::jit::IValue output = updatedModel->forward({validationInput});
    if (!output.isTuple()) {
        throw std::invalid_argument("Updated inference model must return a tuple");
    }
    const auto outputTuple = output.toTuple();
    if (outputTuple->elements().size() != 2 || !outputTuple->elements()[0].isTensor() ||
        !outputTuple->elements()[1].isTensor()) {
        throw std::invalid_argument("Updated inference model must return policy and WDL tensors");
    }
    const torch::Tensor policy = outputTuple->elements()[0].toTensor();
    const torch::Tensor outcome = outputTuple->elements()[1].toTensor();
    if (policy.dim() != 2 || policy.size(0) != 1 || policy.size(1) != ACTION_SIZE ||
        outcome.dim() != 2 || outcome.size(0) != 1 ||
        outcome.size(1) != static_cast<int64_t>(WDL_OUTPUT_SIZE) ||
        !torch::isfinite(policy).all().item<bool>() ||
        !torch::isfinite(outcome).all().item<bool>() || (policy < 0).any().item<bool>() ||
        (outcome < 0).any().item<bool>() || std::abs(policy.sum().item<float>() - 1.0F) > 1e-2F ||
        std::abs(outcome.sum().item<float>() - 1.0F) > 1e-2F) {
        throw std::invalid_argument("Updated inference model returned invalid output");
    }
    return updatedModel;
}

inline void commitInferenceModelUpdate(PreparedInferenceModel &model,
                                       PreparedInferenceModel updatedModel) noexcept {
    assert(model != nullptr);
    assert(updatedModel != nullptr);
    model.swap(updatedModel);
}
