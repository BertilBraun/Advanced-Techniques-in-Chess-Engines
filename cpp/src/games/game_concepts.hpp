#pragma once

#include "common.hpp"

#include <concepts>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace az::games {

template <typename State>
concept GameState =
    std::copy_constructible<State> &&
    requires(State state, const State constant_state, typename State::action_type action) {
        typename State::action_type;
        typename State::player_type;
        typename State::termination_reason_type;
        typename State::terminal_result_type;
        typename State::encoding_type;
        { constant_state.actionCount() } -> std::same_as<typename State::action_type>;
        { constant_state.legalActions() } -> std::same_as<std::vector<typename State::action_type>>;
        { constant_state.isLegal(action) } -> std::same_as<bool>;
        { state.apply(action) } -> std::same_as<void>;
        { constant_state.currentPlayer() } -> std::same_as<typename State::player_type>;
        {
            constant_state.terminationReason()
        } -> std::same_as<typename State::termination_reason_type>;
        { constant_state.isTerminal() } -> std::same_as<bool>;
        { constant_state.terminalResult() } -> std::same_as<typename State::terminal_result_type>;
        { constant_state.canonicalEncoding() } -> std::same_as<typename State::encoding_type>;
        { constant_state.stateHash() } -> std::same_as<uint64>;
        { constant_state == constant_state } -> std::same_as<bool>;
    };

template <typename Operations>
concept GameSymmetry = requires(typename Operations::action_type action,
                                const typename Operations::encoding_type &encoding,
                                typename Operations::symmetry_type symmetry, int32 boardSize) {
    typename Operations::action_type;
    typename Operations::encoding_type;
    typename Operations::symmetry_type;
    { Operations::inverse(symmetry) } -> std::same_as<typename Operations::symmetry_type>;
    {
        Operations::transformAction(action, boardSize, symmetry)
    } -> std::same_as<typename Operations::action_type>;
    {
        Operations::transformEncoding(encoding, symmetry)
    } -> std::same_as<typename Operations::encoding_type>;
};

} // namespace az::games
