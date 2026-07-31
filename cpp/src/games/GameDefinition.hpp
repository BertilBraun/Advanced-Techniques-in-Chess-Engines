#pragma once

#include "common.hpp"
#include "core/ArtifactMetadata.hpp"
#include "games/game_concepts.hpp"

#include <concepts>
#include <optional>
#include <span>
#include <string_view>

namespace az::games {

template <typename Definition>
concept GameDefinition =
    GameState<typename Definition::State> &&
    std::same_as<typename Definition::Action, typename Definition::State::action_type> &&
    std::same_as<typename Definition::Player, typename Definition::State::player_type> &&
    std::same_as<typename Definition::TerminationReason,
                 typename Definition::State::termination_reason_type> &&
    std::same_as<typename Definition::TerminalResult,
                 typename Definition::State::terminal_result_type> &&
    std::same_as<typename Definition::Encoding, typename Definition::State::encoding_type> &&
    std::copy_constructible<typename Definition::Rules> &&
    requires(const typename Definition::Rules &rules, const typename Definition::State &state,
             typename Definition::Action action, typename Definition::Symmetry symmetry,
             int32 policyIndex) {
        typename Definition::ReplayPayload;
        typename Definition::Symmetry;
        { Definition::identity() } -> std::same_as<core::GameIdentity>;
        { Definition::gameSchemaVersion() } -> std::same_as<uint32>;
        { Definition::replaySchemaVersion() } -> std::same_as<uint32>;
        { Definition::policySpaceIdentity() } -> std::same_as<std::string_view>;
        { Definition::createInitialState(rules) } -> std::same_as<typename Definition::State>;
        { Definition::actionCount(rules) } -> std::same_as<int32>;
        { Definition::inputSpecification(rules) } -> std::same_as<core::TensorSpecification>;
        { Definition::actionToPolicy(action, rules) } -> std::same_as<int32>;
        {
            Definition::policyToAction(policyIndex, rules)
        } -> std::same_as<typename Definition::Action>;
        {
            Definition::validSymmetries()
        } -> std::same_as<std::span<const typename Definition::Symmetry>>;
        {
            Definition::transformAction(action, rules, symmetry)
        } -> std::same_as<typename Definition::Action>;
        {
            Definition::transformEncoding(state.canonicalEncoding(), symmetry)
        } -> std::same_as<typename Definition::Encoding>;
        { Definition::replayPayload(state) } -> std::same_as<typename Definition::ReplayPayload>;
        { Definition::terminalValue(state) } -> std::same_as<std::optional<double>>;
    };

} // namespace az::games
