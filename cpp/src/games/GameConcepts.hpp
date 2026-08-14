#pragma once

#include "search/InferenceDimensions.hpp"

#include <concepts>
#include <cstdint>
#include <vector>

template <typename Game>
concept GameRules =
    std::copy_constructible<typename Game::State> &&
    std::copy_constructible<typename Game::Action> &&
    requires(const typename Game::State &state, const typename Game::Action action) {
        { Game::childState(state, action) } -> std::same_as<typename Game::State>;
        { Game::legalActions(state) } -> std::same_as<std::vector<typename Game::Action>>;
        { Game::isTerminal(state) } -> std::same_as<bool>;
        { Game::terminalValue(state) } -> std::same_as<float>;
    };

template <typename Game>
concept GameEncoding = requires(const typename Game::State &state,
                                const typename Game::Action action, std::int8_t *destination) {
    typename Game::Encoding;
    { Game::Encoding::inferenceDimensions() } -> std::same_as<InferenceDimensions>;
    { Game::Encoding::actionId(action, state) } -> std::same_as<int>;
    { Game::Encoding::encodeInputInto(state, destination) } -> std::same_as<void>;
};

template <typename Game>
concept SearchGame = GameRules<Game> && GameEncoding<Game>;
