#pragma once

#include "common.hpp"

using PolicyValue = std::pair<torch::Tensor, torch::Tensor>;

struct ResBlockImpl : torch::nn::Cloneable<ResBlockImpl> {
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm2d bn1;
    torch::nn::Conv2d conv2;
    torch::nn::BatchNorm2d bn2;

    ResBlockImpl(int num_hidden)
        : conv1(torch::nn::Conv2dOptions(num_hidden, num_hidden, 3).padding(1)), bn1(num_hidden),
          conv2(torch::nn::Conv2dOptions(num_hidden, num_hidden, 3).padding(1)), bn2(num_hidden) {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
    }

    torch::Tensor forward(torch::Tensor x) {
        auto residual = x;
        x = torch::relu(bn1(conv1(x)));
        x = bn2(conv2(x));
        x += residual;
        x = torch::relu(x);
        return x;
    }

    void reset() override {}
};
TORCH_MODULE(ResBlock);

struct NetworkImpl : torch::nn::Cloneable<NetworkImpl> {
    torch::Device device = torch::kCUDA;
    torch::nn::Sequential startBlock, policyHead, valueHead;
    torch::nn::ModuleList backBone;

    NetworkImpl() {
        // Set device based on CUDA availability
        device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

        // Initialize start block
        startBlock = torch::nn::Sequential(
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(ENCODING_CHANNELS, NUM_HIDDEN, 3).padding(1)),
            torch::nn::BatchNorm2d(NUM_HIDDEN), torch::nn::Functional(torch::relu));

        // Initialize residual blocks
        for (int i = 0; i < NUM_RES_BLOCKS; ++i) {
            backBone->push_back(ResBlock(NUM_HIDDEN));
        }

        // Initialize policy head
        policyHead = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(NUM_HIDDEN, 32, 3).padding(1)),
            torch::nn::BatchNorm2d(32), torch::nn::Functional(torch::relu), torch::nn::Flatten(),
            torch::nn::Linear(32 * ROW_COUNT * COLUMN_COUNT, ACTION_SIZE));

        // Initialize value head
        valueHead = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(NUM_HIDDEN, 3, 3).padding(1)),
            torch::nn::BatchNorm2d(3), torch::nn::Functional(torch::relu), torch::nn::Flatten(),
            torch::nn::Linear(3 * ROW_COUNT * COLUMN_COUNT, 1), torch::nn::Functional(torch::tanh));

        // Register modules
        register_module("startBlock", startBlock);
        register_module("backBone", backBone);
        register_module("policyHead", policyHead);
        register_module("valueHead", valueHead);

        this->to(device);
    }

    void reset() override {}

    PolicyValue forward(torch::Tensor x) {
        x = startBlock->forward(x);
        for (const auto &block : *backBone) {
            x = block->as<ResBlock>()->forward(x);
        }
        auto policy = policyHead->forward(x);
        auto value = valueHead->forward(x);
        return {policy, value};
    }

    PolicyValue inference(torch::Tensor x) {
        auto result = this->forward(x);
        auto policy = torch::softmax(result.first, 1).to(torch::kCPU).detach().clone();
        auto value = result.second.squeeze(1).to(torch::kCPU).detach().clone();
        return {policy, value};
    }
};
TORCH_MODULE(Network);
