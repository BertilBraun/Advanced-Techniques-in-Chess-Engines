#pragma once

#include "BitBoard.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

template <std::size_t BoardSize, std::size_t BinaryPlaneCount, std::size_t ScalarPlaneCount>
struct PackedPlaneLayout {
    static constexpr std::size_t bit_count = BoardSize * BoardSize;
    static constexpr std::size_t word_count = BitBoard<BoardSize>::word_count;
    static constexpr std::size_t binary_plane_count = BinaryPlaneCount;
    static constexpr std::size_t scalar_plane_count = ScalarPlaneCount;
    static constexpr std::size_t binary_bytes = BinaryPlaneCount * word_count * sizeof(std::uint64_t);
    static constexpr std::size_t payload_bytes = binary_bytes + ScalarPlaneCount * sizeof(std::int8_t);
};

template <std::size_t BoardSize, std::size_t BinaryPlaneCount>
constexpr void serialize_binary_planes(
    const std::array<BitBoard<BoardSize>, BinaryPlaneCount> &planes,
    std::span<std::int8_t> destination) noexcept {
    constexpr std::size_t expected_bytes =
        PackedPlaneLayout<BoardSize, BinaryPlaneCount, 0>::binary_bytes;
    assert(destination.size() == expected_bytes);
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
    constexpr std::size_t expected_bytes =
        PackedPlaneLayout<BoardSize, BinaryPlaneCount, 0>::binary_bytes;
    assert(source.size() == expected_bytes);
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
