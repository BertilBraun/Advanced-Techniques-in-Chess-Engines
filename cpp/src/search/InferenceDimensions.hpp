#pragma once

#include <cstddef>

struct InferenceDimensions {
    std::size_t channels;
    std::size_t rows;
    std::size_t columns;
    std::size_t actions;
    std::size_t outcomes;

    [[nodiscard]] constexpr std::size_t encodedSize() const noexcept {
        return channels * rows * columns;
    }
    [[nodiscard]] bool operator==(const InferenceDimensions &) const noexcept = default;
};
