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
    std::size_t incoming_edges = 0;

    [[nodiscard]] bool expanded() const noexcept { return !children.empty(); }
};

struct GameSearchPathStep {
    std::size_t node_index;
    std::size_t edge_index;
};

struct GameSearchPath {
    std::vector<GameSearchPathStep> steps;
    std::size_t leaf_index;
};

struct GraphSearchSelection {
    GameSearchPath path;
    std::optional<float> immediate_value;
    bool update_leaf;

    [[nodiscard]] bool requiresInference() const noexcept { return !immediate_value.has_value(); }
};

struct GraphSearchStatistics {
    std::uint64_t transposition_table_probes = 0;
    std::uint64_t transposition_table_hits = 0;
    std::uint64_t transposition_links = 0;
    std::uint64_t unique_nodes_created = 0;
    std::uint64_t edges_created = 0;
    std::uint64_t evaluations_avoided = 0;
    std::uint64_t transposition_corrections = 0;
    std::uint64_t correction_clips = 0;
    std::uint64_t continued_transpositions = 0;
    std::uint64_t cycle_cutoffs = 0;
    std::uint64_t nodes_retained = 0;
    std::uint64_t nodes_reclaimed = 0;
    std::uint64_t edges_reclaimed = 0;
    std::uint64_t nodes_pruned = 0;
    std::uint64_t hash_collision_checks = 0;
    std::uint64_t peak_live_nodes = 0;
    std::uint64_t peak_live_edges = 0;
    std::uint64_t identity_lookup_nanoseconds = 0;
    std::uint64_t reroot_nanoseconds = 0;
    std::uint64_t pruning_nanoseconds = 0;

    [[nodiscard]] bool operator==(const GraphSearchStatistics &) const noexcept = default;
};
