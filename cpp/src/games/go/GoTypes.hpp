#pragma once

#include "util/BitBoard.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>

template <std::size_t BoardSize> struct GoBoard {
    BitBoard<BoardSize> black;
    BitBoard<BoardSize> white;

    [[nodiscard]] bool operator==(const GoBoard &) const noexcept = default;
};

template <std::size_t BoardSize> struct GoAction {
    using Point = typename BitBoard<BoardSize>::Point;

    static constexpr int pass_id = static_cast<int>(BoardSize * BoardSize);
    static constexpr int action_count = pass_id + 1;

    int id;

    explicit constexpr GoAction(const int actionId) : id(actionId) {
        if (actionId < 0 || actionId >= action_count) {
            throw std::invalid_argument("Go action id is outside the action space");
        }
    }

    [[nodiscard]] static constexpr GoAction pass() { return GoAction(pass_id); }
    [[nodiscard]] constexpr bool is_pass() const noexcept { return id == pass_id; }
    [[nodiscard]] constexpr Point point() const noexcept {
        assert(!is_pass());
        return BitBoard<BoardSize>::point(static_cast<std::size_t>(id));
    }
    [[nodiscard]] bool operator==(const GoAction &) const noexcept = default;
};

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
    std::optional<GoAreaScore> score;
    std::optional<GoPlayer> winner;

    [[nodiscard]] bool operator==(const GoTerminalResult &) const noexcept = default;
};
