#pragma once

#include "common.hpp"
#include "core/ArtifactMetadata.hpp"
#include "games/chess/ChessEncoding.hpp"
#include "games/chess/ChessState.hpp"

#include <array>
#include <optional>
#include <span>
#include <string_view>

namespace az::games::chess {

enum class ChessSymmetry : int8 { Identity = 0 };

struct ChessDefinition {
    using State = ChessState;
    using Action = int32;
    using Rules = ChessRules;
    using Player = chess::Player;
    using TerminalResult = chess::TerminalResult;
    using TerminationReason = chess::TerminationReason;
    using Encoding = ChessEncoding;
    using ReplayPayload = ChessEncoding;
    using Symmetry = ChessSymmetry;

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
    inline static constexpr std::array<Symmetry, 1> SYMMETRIES{Symmetry::Identity};
};

} // namespace az::games::chess
