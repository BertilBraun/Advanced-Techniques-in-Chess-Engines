#pragma once

#include "games/go/GoPosition.hpp"
#include "util/PackedPlane.hpp"
#include "util/py.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>

template <std::size_t BoardSize, std::size_t HistoryLength> struct EncodedGoPosition {
    static constexpr std::size_t binary_plane_count = HistoryLength * 2;
    static constexpr std::size_t scalar_plane_count = 1;
    static constexpr std::size_t packed_binary_bytes =
        packed_binary_plane_bytes<BoardSize, binary_plane_count>;
    static constexpr std::size_t packed_bytes = packed_binary_bytes + scalar_plane_count;

    std::array<BitBoard<BoardSize>, binary_plane_count> binary_planes;
    std::array<std::int8_t, scalar_plane_count> scalar_planes;

    [[nodiscard]] bool operator==(const EncodedGoPosition &) const noexcept = default;
};

template <std::size_t BoardSize, std::size_t HistoryLength>
[[nodiscard]] EncodedGoPosition<BoardSize, HistoryLength>
encode_go_position(const GoPosition<BoardSize, HistoryLength> &position) {
    EncodedGoPosition<BoardSize, HistoryLength> encoded{};
    for (const auto offset : range(HistoryLength)) {
        const GoBoard<BoardSize> &board = position.history()[offset];
        if (position.player() == GoPlayer::black) {
            encoded.binary_planes[offset * 2] = board.black;
            encoded.binary_planes[offset * 2 + 1] = board.white;
        } else {
            encoded.binary_planes[offset * 2] = board.white;
            encoded.binary_planes[offset * 2 + 1] = board.black;
        }
    }
    encoded.scalar_planes[0] = position.player() == GoPlayer::black ? 1 : 0;
    return encoded;
}

template <std::size_t BoardSize, std::size_t HistoryLength>
void write_packed_go_position(const EncodedGoPosition<BoardSize, HistoryLength> &encoded,
                              std::int8_t *destination) {
    if (destination == nullptr) {
        throw std::invalid_argument("Go packed encoding destination is null");
    }
    serialize_binary_planes<BoardSize,
                            EncodedGoPosition<BoardSize, HistoryLength>::binary_plane_count>(
        encoded.binary_planes,
        std::span<std::int8_t>(destination,
                               EncodedGoPosition<BoardSize, HistoryLength>::packed_binary_bytes));
    std::memcpy(destination + EncodedGoPosition<BoardSize, HistoryLength>::packed_binary_bytes,
                encoded.scalar_planes.data(),
                EncodedGoPosition<BoardSize, HistoryLength>::scalar_plane_count);
}

template <std::size_t BoardSize, std::size_t HistoryLength>
void write_go_tensor_encoding(const EncodedGoPosition<BoardSize, HistoryLength> &encoded,
                              std::int8_t *destination) {
    if (destination == nullptr) {
        throw std::invalid_argument("Go tensor encoding destination is null");
    }
    std::size_t offset = 0;
    for (const BitBoard<BoardSize> &plane : encoded.binary_planes) {
        for (const auto point : range(BitBoard<BoardSize>::bit_count)) {
            destination[offset++] = plane.test(point) ? 1 : 0;
        }
    }
    for (const std::int8_t value : encoded.scalar_planes) {
        std::fill_n(destination + offset, BitBoard<BoardSize>::bit_count, value);
        offset += BitBoard<BoardSize>::bit_count;
    }
}

template <std::size_t BoardSize, std::size_t HistoryLength = 8> struct GoRepresentationDimensions {
    static constexpr int board_length = static_cast<int>(BoardSize);
    static constexpr int binary_channel_count = static_cast<int>(HistoryLength * 2);
    static constexpr int scalar_channel_count = 1;
    static constexpr int channel_count = binary_channel_count + scalar_channel_count;
    static constexpr int action_count = static_cast<int>(BoardSize * BoardSize + 1);
};
