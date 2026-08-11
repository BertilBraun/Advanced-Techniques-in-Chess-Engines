#pragma once

#include "games/GameConcepts.hpp"
#include "search/InferenceTypes.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

template <typename Action> struct GameSearchEdge {
    Action action;
    float raw_prior;
    float prior;
    std::uint32_t visits = 0;
    float value_sum = 0.0F;
    float virtual_loss = 0.0F;
    std::optional<std::size_t> child_index;
};

template <SearchGame Game> struct GameSearchNode {
    using Position = typename Game::State;
    using Action = typename Game::Action;

    Position position;
    std::vector<GameSearchEdge<Action>> children;
    std::optional<std::size_t> parent_index;
    std::optional<std::size_t> parent_edge_index;
    std::uint32_t visits = 0;
    float value_sum = 0.0F;
    float virtual_loss = 0.0F;
    bool inference_pending = false;
    std::optional<WdlPrediction> network_outcome;

    [[nodiscard]] bool expanded() const noexcept { return !children.empty(); }
};
