#pragma once

#include <concepts>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace az::v2::games {

template <typename State>
concept GameState =
    std::copy_constructible<State> &&
    requires(State state, const State constant_state, typename State::action_type action) {
        typename State::action_type;
        typename State::player_type;
        typename State::termination_reason_type;
        typename State::terminal_result_type;
        typename State::encoding_type;
        { constant_state.action_count() } -> std::same_as<typename State::action_type>;
        {
            constant_state.legal_actions()
        } -> std::same_as<std::vector<typename State::action_type>>;
        { constant_state.is_legal(action) } -> std::same_as<bool>;
        { state.apply(action) } -> std::same_as<void>;
        { constant_state.current_player() } -> std::same_as<typename State::player_type>;
        {
            constant_state.termination_reason()
        } -> std::same_as<typename State::termination_reason_type>;
        { constant_state.is_terminal() } -> std::same_as<bool>;
        { constant_state.terminal_result() } -> std::same_as<typename State::terminal_result_type>;
        { constant_state.canonical_encoding() } -> std::same_as<typename State::encoding_type>;
        { constant_state.state_hash() } -> std::same_as<std::uint64_t>;
        { constant_state == constant_state } -> std::same_as<bool>;
    };

template <typename Operations>
concept GameSymmetry =
    requires(typename Operations::action_type action,
             const typename Operations::encoding_type &encoding,
             typename Operations::symmetry_type symmetry, std::int32_t board_size) {
        typename Operations::action_type;
        typename Operations::encoding_type;
        typename Operations::symmetry_type;
        { Operations::inverse(symmetry) } -> std::same_as<typename Operations::symmetry_type>;
        {
            Operations::transform_action(action, board_size, symmetry)
        } -> std::same_as<typename Operations::action_type>;
        {
            Operations::transform_encoding(encoding, symmetry)
        } -> std::same_as<typename Operations::encoding_type>;
    };

} // namespace az::v2::games
