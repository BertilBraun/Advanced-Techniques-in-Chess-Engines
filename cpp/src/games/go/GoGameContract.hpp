#pragma once

#include "games/go/GoEncoding.hpp"
#include "games/go/GoPosition.hpp"
#include "search/InferenceTypes.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

template <std::size_t BoardSize, std::size_t HistoryLength = 8> class GoGameContract {
public:
    using Position = GoPosition<BoardSize, HistoryLength>;
    using Action = GoAction<BoardSize>;

    [[nodiscard]] static constexpr InferenceDimensions inferenceDimensions() noexcept {
        return {
            .channels = GoRepresentationDimensions<BoardSize, HistoryLength>::channel_count,
            .rows = GoRepresentationDimensions<BoardSize, HistoryLength>::board_length,
            .columns = GoRepresentationDimensions<BoardSize, HistoryLength>::board_length,
            .actions = GoRepresentationDimensions<BoardSize, HistoryLength>::action_count,
            .outcomes = 3,
        };
    }

    [[nodiscard]] static constexpr float searchTurnDiscount() noexcept { return 1.0F; }

    [[nodiscard]] static Position childPosition(const Position &parent, const Action action) {
        return parent.child(action);
    }
    [[nodiscard]] static std::vector<Action> legalActions(const Position &position) {
        return position.legal_actions();
    }
    [[nodiscard]] static bool isTerminal(const Position &position) noexcept {
        return position.is_terminal();
    }
    [[nodiscard]] static std::optional<float> terminalValue(const Position &position) {
        const GoTerminalResult result = position.terminal_result();
        if (result.reason == GoTerminationReason::ongoing ||
            result.reason == GoTerminationReason::maximum_moves) {
            return std::nullopt;
        }
        if (!result.winner.has_value()) {
            return 0.0F;
        }
        return *result.winner == position.player() ? 1.0F : -1.0F;
    }
    [[nodiscard]] static int actionId(const Action action, const Position &) noexcept {
        return action.id;
    }
    static void encodeInputInto(const Position &position, std::int8_t *destination) {
        write_go_tensor_encoding(encode_go_position(position), destination);
    }
};

using Go7GameContract = GoGameContract<7, 8>;
using Go9GameContract = GoGameContract<9, 8>;
