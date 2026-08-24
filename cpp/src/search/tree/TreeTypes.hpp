#pragma once

#include "games/GameConcepts.hpp"
#include "search/InferenceTypes.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

template <typename Action> struct GameSearchEdge {
    Action action;
    int action_id;
    float raw_prior;
    float prior;
    std::uint32_t visits = 0;
    float value_sum = 0.0F;
    float virtual_loss = 0.0F;
    std::optional<std::size_t> child_index;
};

// The position is last: selection and backup touch only the fields above it, and a chess position is
// over a kilobyte, so leading with it would put every hot field on its own cache line.
template <SearchGame Game> struct GameSearchNode {
    using Position = typename Game::State;
    using Action = typename Game::Action;

    std::vector<GameSearchEdge<Action>> children;
    std::optional<std::size_t> parent_index;
    std::optional<std::size_t> parent_edge_index;
    std::uint32_t visits = 0;
    float value_sum = 0.0F;
    float virtual_loss = 0.0F;
    bool inference_pending = false;
    float search_correction = 0.0F;
    std::optional<WdlPrediction> network_outcome;
    Position position;

    [[nodiscard]] bool expanded() const noexcept { return !children.empty(); }
};
