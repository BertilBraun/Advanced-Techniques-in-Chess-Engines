#pragma once

#include <cstdint>
#include <optional>

enum class GoPlayer : std::uint8_t { black = 1, white = 2 };

enum class GoTerminationReason : std::uint8_t { ongoing = 0, two_passes = 1, maximum_moves = 2 };

struct GoRules {
    int komi_half_points;
    int maximum_moves;

    [[nodiscard]] bool operator==(const GoRules &) const noexcept = default;
};

struct GoAreaScore {
    int black_half_points;
    int white_half_points;

    [[nodiscard]] std::optional<GoPlayer> winner() const noexcept {
        if (black_half_points > white_half_points) {
            return GoPlayer::black;
        }
        if (white_half_points > black_half_points) {
            return GoPlayer::white;
        }
        return std::nullopt;
    }

    [[nodiscard]] bool operator==(const GoAreaScore &) const noexcept = default;
};

struct GoTerminalResult {
    GoTerminationReason reason;
    GoAreaScore score;
    std::optional<GoPlayer> winner;

    [[nodiscard]] bool operator==(const GoTerminalResult &) const noexcept = default;
};
