#pragma once

#include "common.hpp"

#include <string>
#include <vector>

namespace az::core {

enum class GameIdentity : int8 { Go = 0, Chess = 1 };

enum class TensorDataType : int8 { Int8 = 0, Float32 = 1 };

enum class TensorLayout : int8 { ChannelsFirst = 0, Flat = 1 };

struct TensorSpecification {
    TensorDataType dataType;
    TensorLayout layout;
    std::vector<int64> dimensions;

    void validate() const {
        if (dimensions.empty()) {
            throw std::invalid_argument("tensor dimensions must not be empty");
        }
        for (const int64 dimension : dimensions) {
            if (dimension <= 0) {
                throw std::invalid_argument("tensor dimensions must be positive");
            }
        }
    }

    [[nodiscard]] bool operator==(const TensorSpecification &) const = default;
};

struct InferenceArtifactMetadata {
    GameIdentity game;
    uint32 artifactSchemaVersion;
    uint32 gameSchemaVersion;
    uint64 modelGeneration;
    std::string artifactIdentity;
    std::string modelFamily;
    std::string policySpaceIdentity;
    std::string contentChecksum;
    TensorSpecification input;
    TensorSpecification policyOutput;
    TensorSpecification valueOutput;

    void validate() const {
        if (artifactSchemaVersion == 0 || gameSchemaVersion == 0) {
            throw std::invalid_argument("artifact schema versions must be positive");
        }
        if (artifactIdentity.empty() || modelFamily.empty() || policySpaceIdentity.empty() ||
            contentChecksum.empty()) {
            throw std::invalid_argument("artifact identities and checksum must not be empty");
        }
        input.validate();
        policyOutput.validate();
        valueOutput.validate();
    }
};

} // namespace az::core
