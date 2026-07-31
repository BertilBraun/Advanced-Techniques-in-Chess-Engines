#pragma once

#include "common.hpp"
#include "core/ArtifactMetadata.hpp"
#include "games/GameDefinition.hpp"

#include <string>
#include <utility>

namespace az::core {

struct NativeSessionMetadata {
    GameIdentity game;
    uint32 gameSchemaVersion;
    uint32 replaySchemaVersion;
    int32 actionCount;
    TensorSpecification canonicalInput;
    std::string policySpaceIdentity;
    InferenceArtifactMetadata artifact;

    void validate() const {
        artifact.validate();
        canonicalInput.validate();
        if (gameSchemaVersion == 0 || replaySchemaVersion == 0 || actionCount <= 0) {
            throw std::invalid_argument("session schemas and action count must be positive");
        }
        if (policySpaceIdentity.empty()) {
            throw std::invalid_argument("session policy-space identity must not be empty");
        }
        if (artifact.game != game || artifact.gameSchemaVersion != gameSchemaVersion ||
            artifact.policySpaceIdentity != policySpaceIdentity ||
            artifact.input != canonicalInput) {
            throw std::invalid_argument("artifact metadata is incompatible with the game session");
        }
        const TensorSpecification expectedPolicy{
            .dataType = TensorDataType::Float32,
            .layout = TensorLayout::Flat,
            .dimensions = {actionCount},
        };
        const TensorSpecification expectedValue{
            .dataType = TensorDataType::Float32,
            .layout = TensorLayout::Flat,
            .dimensions = {1},
        };
        if (artifact.policyOutput != expectedPolicy || artifact.valueOutput != expectedValue) {
            throw std::invalid_argument("artifact output shapes are incompatible with the session");
        }
    }
};

template <games::GameDefinition Definition> struct ResolvedSessionConfiguration {
    using Rules = typename Definition::Rules;

    Rules rules;
    NativeSessionMetadata metadata;
};

template <games::GameDefinition Definition>
[[nodiscard]] ResolvedSessionConfiguration<Definition>
resolveSessionConfiguration(typename Definition::Rules rules, InferenceArtifactMetadata artifact) {
    const int32 actionCount = Definition::actionCount(rules);
    NativeSessionMetadata metadata{
        .game = Definition::identity(),
        .gameSchemaVersion = Definition::gameSchemaVersion(),
        .replaySchemaVersion = Definition::replaySchemaVersion(),
        .actionCount = actionCount,
        .canonicalInput = Definition::inputSpecification(rules),
        .policySpaceIdentity = std::string(Definition::policySpaceIdentity()),
        .artifact = std::move(artifact),
    };
    metadata.validate();
    return ResolvedSessionConfiguration<Definition>{
        .rules = std::move(rules),
        .metadata = std::move(metadata),
    };
}

} // namespace az::core
