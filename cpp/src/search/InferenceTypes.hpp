#pragma once

#include "games/GameConcepts.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

// Defines the game-independent dimensions, predictions, results, and telemetry of inference.

enum class WdlIndex : std::size_t { Win = 0, Draw = 1, Loss = 2, Count = 3 };
inline constexpr std::size_t WDL_OUTPUT_SIZE = static_cast<std::size_t>(WdlIndex::Count);

// One predicted log-KL value per search-budget grid point.
inline constexpr std::size_t SEARCH_BUDGET_CURVE_POINTS = 10;
using SearchBudgetCurvePrediction = std::array<float, SEARCH_BUDGET_CURVE_POINTS>;

struct WdlPrediction {
    float win;
    float draw;
    float loss;

    [[nodiscard]] float expectedValue() const noexcept { return win - loss; }
    [[nodiscard]] float value() const noexcept { return expectedValue(); }
    [[nodiscard]] bool operator==(const WdlPrediction &) const noexcept = default;
};

using OutcomeProbabilities = WdlPrediction;

// The action id is carried from inference so the search never recomputes it per edge visit.
template <typename Action> struct ScoredAction {
    Action action;
    int action_id;
    float prior;
};

template <SearchGame Game> struct SearchInferenceResult {
    using Action = typename Game::Action;

    std::vector<ScoredAction<Action>> actions;
    WdlPrediction outcome;
    SearchBudgetCurvePrediction search_budget_curve;

    [[nodiscard]] float value() const noexcept { return outcome.expectedValue(); }
};

enum class InferenceDevice { Auto, Cpu, Cuda };
enum class SdpaBackend { Automatic, Flash, MemoryEfficient, Math, CuDNN };

// CUDA inference precision. CPU inference stays float32 whatever is requested here.
enum class InferencePrecision { BFloat16, Float16, Float32 };

// cuDNN reaches its tensor-core convolution kernels only through channels-last activations, so the
// layout is switchable independently of the precision.
enum class InferenceMemoryFormat { Contiguous, ChannelsLast };

// Every default reproduces the shipped bfloat16, contiguous, heuristic-cuDNN path exactly.
struct InferenceExecutionOptions {
    SdpaBackend sdpa_backend = SdpaBackend::Automatic;
    InferencePrecision precision = InferencePrecision::BFloat16;
    InferenceMemoryFormat memory_format = InferenceMemoryFormat::Contiguous;
    bool cudnn_benchmark = false;

    [[nodiscard]] bool operator==(const InferenceExecutionOptions &) const noexcept = default;
};

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
