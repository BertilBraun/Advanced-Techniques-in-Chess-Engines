#pragma once

#include "games/go/implementation/GoPosition.hpp"
#include "search/InferenceTypes.hpp"
#include "util/PackedPlane.hpp"
#include "util/py.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

template <std::size_t BoardSize, std::size_t HistoryLength>
using EncodedGoPosition = EncodedPlanes<BoardSize, HistoryLength * 2, 1>;

template <std::size_t BoardSize, std::size_t HistoryLength>
[[nodiscard]] EncodedGoPosition<BoardSize, HistoryLength>
encodeGoPosition(const GoPosition<BoardSize, HistoryLength> &position) {
    EncodedGoPosition<BoardSize, HistoryLength> encoded{};
    for (const auto offset : range(HistoryLength)) {
        const GoBoard<BoardSize> &board = position.history()[offset];
        if (position.player() == GoPlayer::black) {
            encoded.binaryPlanes[offset * 2] = board.black;
            encoded.binaryPlanes[offset * 2 + 1] = board.white;
        } else {
            encoded.binaryPlanes[offset * 2] = board.white;
            encoded.binaryPlanes[offset * 2 + 1] = board.black;
        }
    }
    encoded.scalarPlanes[0] = position.player() == GoPlayer::black ? 1 : 0;
    return encoded;
}

template <std::size_t BoardSize, std::size_t HistoryLength = 8> struct GoRepresentationDimensions {
    static constexpr int boardLength = static_cast<int>(BoardSize);
    static constexpr int binaryChannelCount = static_cast<int>(HistoryLength * 2);
    static constexpr int scalarChannelCount = 1;
    static constexpr int channelCount = binaryChannelCount + scalarChannelCount;
    static constexpr int actionCount = static_cast<int>(BoardSize * BoardSize + 1);
};

template <std::size_t BoardSize, std::size_t HistoryLength> struct GoEncoding {
    using State = GoPosition<BoardSize, HistoryLength>;
    using Action = GoAction<BoardSize>;

    [[nodiscard]] static constexpr InferenceDimensions inferenceDimensions() noexcept {
        return {
            .channels = GoRepresentationDimensions<BoardSize, HistoryLength>::channelCount,
            .rows = GoRepresentationDimensions<BoardSize, HistoryLength>::boardLength,
            .columns = GoRepresentationDimensions<BoardSize, HistoryLength>::boardLength,
            .actions = GoRepresentationDimensions<BoardSize, HistoryLength>::actionCount,
            .outcomes = WDL_OUTPUT_SIZE,
        };
    }

    [[nodiscard]] static int actionId(const Action action, const State &) noexcept {
        return action.id;
    }

    [[nodiscard]] static Action decodeAction(const int actionId, const State &) {
        return Action(actionId);
    }

    static void encodeInputInto(const State &state, std::int8_t *destination) {
        const EncodedGoPosition<BoardSize, HistoryLength> encoded = encodeGoPosition(state);
        encoded.writeTensorInto(std::span<std::int8_t>(
            destination, EncodedGoPosition<BoardSize, HistoryLength>::tensorValues));
    }
};
