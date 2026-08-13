#pragma once

#include "games/GameConcepts.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

// Defines the game-independent dimensions, predictions, results, and telemetry of inference.

enum class WdlIndex : std::size_t { Win = 0, Draw = 1, Loss = 2, Count = 3 };
inline constexpr std::size_t WDL_OUTPUT_SIZE = static_cast<std::size_t>(WdlIndex::Count);

struct WdlPrediction {
    float win;
    float draw;
    float loss;

    [[nodiscard]] float expectedValue() const noexcept { return win - loss; }
    [[nodiscard]] float value() const noexcept { return expectedValue(); }
    [[nodiscard]] bool operator==(const WdlPrediction &) const noexcept = default;
};

using OutcomeProbabilities = WdlPrediction;

template <SearchGame Game> struct SearchInferenceResult {
    using Action = typename Game::Action;

    std::vector<std::pair<Action, float>> actions;
    WdlPrediction outcome;

    [[nodiscard]] float value() const noexcept { return outcome.expectedValue(); }
};

enum class InferenceDevice { Auto, Cpu, Cuda };

struct InferenceStatistics {
    std::size_t evaluations = 0;
    std::size_t modelInferenceCalls = 0;
    std::size_t modelInferencePositions = 0;
    std::vector<std::size_t> modelBatchSizeHistogram;
    float averageNumberOfPositionsInInferenceCall = 0.0F;
    std::uint64_t treeSelectionNanoseconds = 0;
    std::uint64_t boardEncodingNanoseconds = 0;
    std::uint64_t resultProcessingNanoseconds = 0;
    std::uint64_t treeBackupNanoseconds = 0;
    std::uint64_t treeOwnerWaitNanoseconds = 0;
    // CUDA inference time measures host submission; event synchronization is tree-owner wait.
    std::uint64_t inferenceNanoseconds = 0;
    float workerUtilization = 0.0F;
};
