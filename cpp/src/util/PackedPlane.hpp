#pragma once

#include "BitBoard.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

template <std::size_t BoardSize, std::size_t BinaryPlaneCount>
inline constexpr std::size_t packed_binary_plane_bytes =
    BinaryPlaneCount * BitBoard<BoardSize>::word_count * sizeof(std::uint64_t);

template <std::size_t BoardSize, std::size_t BinaryPlaneCount>
constexpr void serialize_binary_planes(
    const std::array<BitBoard<BoardSize>, BinaryPlaneCount> &planes,
    std::span<std::int8_t> destination) noexcept {
    assert((destination.size() == packed_binary_plane_bytes<BoardSize, BinaryPlaneCount>));
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
    assert((source.size() == packed_binary_plane_bytes<BoardSize, BinaryPlaneCount>));
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
