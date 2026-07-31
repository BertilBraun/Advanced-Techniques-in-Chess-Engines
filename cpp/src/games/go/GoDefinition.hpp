#pragma once

#include "common.hpp"
#include "core/ArtifactMetadata.hpp"
#include "games/go/GoEncoding.hpp"
#include "games/go/GoState.hpp"
#include "games/go/GoSymmetry.hpp"

#include <array>
#include <optional>
#include <span>
#include <string_view>

namespace az::games::go {

struct GoDefinition {
    using State = GoState;
    using Action = int32;
    using Rules = GoRules;
    using Player = go::Player;
    using TerminalResult = go::TerminalResult;
    using TerminationReason = go::TerminationReason;
    using Encoding = GoEncoding;
    using ReplayPayload = GoEncoding;
    using Symmetry = go::Symmetry;

    [[nodiscard]] static core::GameIdentity identity();
    [[nodiscard]] static uint32 gameSchemaVersion();
    [[nodiscard]] static uint32 replaySchemaVersion();
    [[nodiscard]] static std::string_view policySpaceIdentity();
    [[nodiscard]] static State createInitialState(const Rules &rules);
    [[nodiscard]] static int32 actionCount(const Rules &rules);
    [[nodiscard]] static core::TensorSpecification inputSpecification(const Rules &rules);
    [[nodiscard]] static int32 actionToPolicy(Action action, const Rules &rules);
    [[nodiscard]] static Action policyToAction(int32 policyIndex, const Rules &rules);
    [[nodiscard]] static std::span<const Symmetry> validSymmetries();
    [[nodiscard]] static Action transformAction(Action action, const Rules &rules,
                                                Symmetry symmetry);
    [[nodiscard]] static Encoding transformEncoding(const Encoding &encoding, Symmetry symmetry);
    [[nodiscard]] static ReplayPayload replayPayload(const State &state);
    [[nodiscard]] static std::optional<double> terminalValue(const State &state);

private:
    inline static constexpr std::array<Symmetry, 8> SYMMETRIES{
        Symmetry::Identity,         Symmetry::Rotate90,         Symmetry::Rotate180,
        Symmetry::Rotate270,        Symmetry::Reflect,          Symmetry::ReflectRotate90,
        Symmetry::ReflectRotate180, Symmetry::ReflectRotate270,
    };
};

} // namespace az::games::go
