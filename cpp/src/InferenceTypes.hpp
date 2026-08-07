#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

enum class InferenceDevice { Auto, Cpu, Cuda };

struct InferenceDimensions {
    int channels;
    int rows;
    int columns;
    int actions;
    int outcomes;

    [[nodiscard]] bool operator==(const InferenceDimensions &) const noexcept = default;
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
    std::uint64_t directInferenceNanoseconds = 0;
    float directWorkerUtilization = 0.0F;
};
