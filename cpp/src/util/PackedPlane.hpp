#pragma once

#include "BitBoard.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

template <std::size_t BoardSize, std::size_t BinaryPlaneCount>
inline constexpr std::size_t packed_binary_plane_bytes =
    BinaryPlaneCount * BitBoard<BoardSize>::word_count * sizeof(std::uint64_t);

template <std::size_t BoardSize, std::size_t BinaryPlaneCount>
constexpr void
serialize_binary_planes(const std::array<BitBoard<BoardSize>, BinaryPlaneCount> &planes,
                        std::span<std::int8_t> destination) noexcept {
    assert((destination.size() == packed_binary_plane_bytes<BoardSize, BinaryPlaneCount>) );
    std::size_t offset = 0;
    for (const BitBoard<BoardSize> &plane : planes) {
        for (std::uint64_t word : plane.words()) {
            std::memcpy(destination.data() + offset, &word, sizeof(word));
            offset += sizeof(word);
        }
    }
}

template <std::size_t BoardSize, std::size_t BinaryPlaneCount>
[[nodiscard]]
constexpr std::array<BitBoard<BoardSize>, BinaryPlaneCount>
deserialize_binary_planes(std::span<const std::int8_t> source) noexcept {
    assert((source.size() == packed_binary_plane_bytes<BoardSize, BinaryPlaneCount>) );
    std::array<BitBoard<BoardSize>, BinaryPlaneCount> planes{};
    std::size_t offset = 0;
    for (BitBoard<BoardSize> &plane : planes) {
        typename BitBoard<BoardSize>::Storage words{};
        for (std::uint64_t &word : words) {
            std::memcpy(&word, source.data() + offset, sizeof(word));
            offset += sizeof(word);
        }
        plane = BitBoard<BoardSize>(words);
    }
    return planes;
}

template <std::size_t BoardSize, std::size_t BinaryPlaneCount, std::size_t ScalarPlaneCount>
struct EncodedPlanes {
    static constexpr std::size_t binary_plane_count = BinaryPlaneCount;
    static constexpr std::size_t scalar_plane_count = ScalarPlaneCount;
    static constexpr std::size_t packed_binary_bytes =
        packed_binary_plane_bytes<BoardSize, BinaryPlaneCount>;
    static constexpr std::size_t packed_bytes = packed_binary_bytes + ScalarPlaneCount;
    static constexpr std::size_t tensor_values =
        (BinaryPlaneCount + ScalarPlaneCount) * BitBoard<BoardSize>::bit_count;

    std::array<BitBoard<BoardSize>, BinaryPlaneCount> binary_planes;
    std::array<std::int8_t, ScalarPlaneCount> scalar_planes;

    void writePackedInto(std::span<std::int8_t> destination) const noexcept {
        assert(destination.size() == packed_bytes);
        serialize_binary_planes<BoardSize, BinaryPlaneCount>(
            binary_planes, destination.first(packed_binary_bytes));
        std::memcpy(destination.data() + packed_binary_bytes, scalar_planes.data(),
                    ScalarPlaneCount);
    }

    void writeTensorInto(std::span<std::int8_t> destination) const noexcept {
        assert(destination.size() == tensor_values);
        std::size_t offset = 0;
        for (const BitBoard<BoardSize> &plane : binary_planes) {
            for (std::size_t firstPoint = 0; firstPoint < BitBoard<BoardSize>::bit_count;
                 firstPoint += 8) {
                const std::uint8_t byte =
                    static_cast<std::uint8_t>(plane.word(firstPoint / 64) >> (firstPoint % 64));
                const std::size_t pointsInByte =
                    std::min<std::size_t>(8, BitBoard<BoardSize>::bit_count - firstPoint);
                for (std::size_t bit = 0; bit < pointsInByte; ++bit) {
                    destination[offset++] = static_cast<std::int8_t>((byte >> bit) & 1U);
                }
            }
        }
        for (const std::int8_t value : scalar_planes) {
            std::fill_n(destination.data() + offset, BitBoard<BoardSize>::bit_count, value);
            offset += BitBoard<BoardSize>::bit_count;
        }
    }

    [[nodiscard]] bool operator==(const EncodedPlanes &) const noexcept = default;
};
