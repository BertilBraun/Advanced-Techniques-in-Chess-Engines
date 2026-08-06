#pragma once

#include "games/go/GoAction.hpp"
#include "games/go/GoEncoding.hpp"
#include "games/go/GoPosition.hpp"
#include "games/go/GoSymmetry.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

template <std::size_t BoardSize, std::size_t HistoryLength = 8> class GoGameContract {
public:
    using Position = GoPosition<BoardSize, HistoryLength>;
    using Action = GoAction<BoardSize>;
    using EncodedPosition = EncodedGoPosition<BoardSize, HistoryLength>;
    using Point = typename Position::Point;

    [[nodiscard]] static Position initialPosition(const GoRules rules) { return Position(rules); }
    [[nodiscard]] static Position childPosition(const Position &parent, const Action action) {
        return parent.child(action);
    }
    [[nodiscard]] static std::vector<Action> legalActions(const Position &position) {
        return position.legal_actions();
    }
    [[nodiscard]] static bool isTerminal(const Position &position) noexcept {
        return position.is_terminal();
    }
    [[nodiscard]] static GoTerminalResult terminalResult(const Position &position) {
        return position.terminal_result();
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
    [[nodiscard]] static std::vector<Action> decodeActions(const std::vector<int> &action_ids,
                                                           const Position &) {
        std::vector<Action> actions;
        actions.reserve(action_ids.size());
        for (const int action_id : action_ids) {
            actions.emplace_back(action_id);
        }
        return actions;
    }
    [[nodiscard]] static EncodedPosition encodeInput(const Position &position) {
        return encode_go_position(position);
    }
    static void writePackedInput(const EncodedPosition &encoded, std::int8_t *destination) {
        write_packed_go_position(encoded, destination);
    }
    static void encodeInputInto(const Position &position, std::int8_t *destination) {
        write_go_tensor_encoding(encodeInput(position), destination);
    }
    [[nodiscard]] static Point transformPoint(const Point point, const GoSymmetry symmetry) {
        return transform_go_point<BoardSize>(point, symmetry);
    }
    [[nodiscard]] static GoSymmetry inverseSymmetry(const GoSymmetry symmetry) {
        return inverse_go_symmetry(symmetry);
    }
    [[nodiscard]] static Action transformAction(const Action action, const GoSymmetry symmetry) {
        return transform_go_action(action, symmetry);
    }
    [[nodiscard]] static EncodedPosition transformEncoding(const EncodedPosition &encoding,
                                                           const GoSymmetry symmetry) {
        return transform_go_encoding(encoding, symmetry);
    }
};

using Go7GameContract = GoGameContract<7, 8>;
using Go9GameContract = GoGameContract<9, 8>;
