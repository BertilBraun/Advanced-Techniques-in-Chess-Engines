#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace az::v2::search {

inline constexpr std::int32_t maximum_simulation_count = std::numeric_limits<std::int32_t>::max();

struct RootNoiseConfiguration {
    bool enabled;
    double alpha;
    double fraction;
};

struct FixedPuctConfiguration {
    std::int32_t simulation_cap;
    double exploration_constant;
    double backup_discount;
    double no_visited_child_value;
    double action_temperature;
    std::uint64_t root_noise_seed;
    std::uint64_t action_sampling_seed;
    RootNoiseConfiguration root_noise;
    bool tree_reuse;

    void validate() const {
        if (simulation_cap <= 0) {
            throw std::invalid_argument("simulation_cap must be positive");
        }
        if (!std::isfinite(exploration_constant) || !(exploration_constant > 0.0)) {
            throw std::invalid_argument("exploration_constant must be finite and positive");
        }
        if (!std::isfinite(backup_discount) || !(backup_discount > 0.0 && backup_discount <= 1.0)) {
            throw std::invalid_argument("backup_discount must be finite and in (0, 1]");
        }
        if (!std::isfinite(no_visited_child_value) ||
            !(no_visited_child_value >= -1.0 && no_visited_child_value <= 1.0)) {
            throw std::invalid_argument("no_visited_child_value must be finite and in [-1, 1]");
        }
        if (!std::isfinite(action_temperature) || !(action_temperature >= 0.0)) {
            throw std::invalid_argument("action_temperature must be finite and non-negative");
        }
        if (!std::isfinite(root_noise.alpha) || !(root_noise.alpha > 0.0) ||
            !std::isfinite(root_noise.fraction) ||
            !(root_noise.fraction >= 0.0 && root_noise.fraction <= 1.0)) {
            throw std::invalid_argument(
                "root noise requires finite positive alpha and finite fraction in [0, 1]");
        }
        if (tree_reuse) {
            throw std::invalid_argument("tree reuse is not implemented for fixed PUCT");
        }
    }
};

} // namespace az::v2::search
