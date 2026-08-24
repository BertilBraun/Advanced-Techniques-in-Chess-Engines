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
inline constexpr std::size_t PACKED_BINARY_PLANE_BYTES =
    BinaryPlaneCount * BitBoard<BoardSize>::wordCount * sizeof(std::uint64_t);

template <std::size_t BoardSize, std::size_t BinaryPlaneCount>
constexpr void
serializeBinaryPlanes(const std::array<BitBoard<BoardSize>, BinaryPlaneCount> &planes,
                      std::span<std::int8_t> destination) noexcept {
    assert((destination.size() == PACKED_BINARY_PLANE_BYTES<BoardSize, BinaryPlaneCount>) );
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
deserializeBinaryPlanes(std::span<const std::int8_t> source) noexcept {
    assert((source.size() == PACKED_BINARY_PLANE_BYTES<BoardSize, BinaryPlaneCount>) );
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

// Byte -> eight little-endian 0/1 bytes, so a packed plane expands with one store per byte.
inline constexpr std::array<std::uint64_t, 256> EXPANDED_BITS = [] {
    std::array<std::uint64_t, 256> table{};
    for (std::size_t byte = 0; byte < table.size(); ++byte) {
        std::uint64_t expanded = 0;
        for (std::size_t bit = 0; bit < 8; ++bit) {
            expanded |= static_cast<std::uint64_t>((byte >> bit) & 1U) << (8 * bit);
        }
        table[byte] = expanded;
    }
    return table;
}();

template <std::size_t BoardSize, std::size_t BinaryPlaneCount, std::size_t ScalarPlaneCount>
struct EncodedPlanes {
    static constexpr std::size_t binaryPlaneCount = BinaryPlaneCount;
    static constexpr std::size_t scalarPlaneCount = ScalarPlaneCount;
    static constexpr std::size_t packedBinaryBytes =
        PACKED_BINARY_PLANE_BYTES<BoardSize, BinaryPlaneCount>;
    static constexpr std::size_t packedBytes = packedBinaryBytes + ScalarPlaneCount;
    static constexpr std::size_t tensorValues =
        (BinaryPlaneCount + ScalarPlaneCount) * BitBoard<BoardSize>::bitCount;

    std::array<BitBoard<BoardSize>, BinaryPlaneCount> binaryPlanes;
    std::array<std::int8_t, ScalarPlaneCount> scalarPlanes;

    void writePackedInto(std::span<std::int8_t> destination) const noexcept {
        assert(destination.size() == packedBytes);
        serializeBinaryPlanes<BoardSize, BinaryPlaneCount>(binaryPlanes,
                                                           destination.first(packedBinaryBytes));
        std::memcpy(destination.data() + packedBinaryBytes, scalarPlanes.data(), ScalarPlaneCount);
    }

    void writeTensorInto(std::span<std::int8_t> destination) const noexcept {
        assert(destination.size() == tensorValues);
        std::size_t offset = 0;
        for (const BitBoard<BoardSize> &plane : binaryPlanes) {
            for (std::size_t firstPoint = 0; firstPoint < BitBoard<BoardSize>::bitCount;
                 firstPoint += 8) {
                const std::uint8_t byte =
                    static_cast<std::uint8_t>(plane.word(firstPoint / 64) >> (firstPoint % 64));
                const std::size_t pointsInByte =
                    std::min<std::size_t>(8, BitBoard<BoardSize>::bitCount - firstPoint);
                // One widened store per byte; the scalar bit loop was a tenth of self-play encoding.
                const std::uint64_t expanded = EXPANDED_BITS[byte];
                if (pointsInByte == 8) {
                    std::memcpy(destination.data() + offset, &expanded, sizeof(expanded));
                } else {
                    std::memcpy(destination.data() + offset, &expanded, pointsInByte);
                }
                offset += pointsInByte;
            }
        }
        for (const std::int8_t value : scalarPlanes) {
            std::fill_n(destination.data() + offset, BitBoard<BoardSize>::bitCount, value);
            offset += BitBoard<BoardSize>::bitCount;
        }
    }

    [[nodiscard]] bool operator==(const EncodedPlanes &) const noexcept = default;
};
