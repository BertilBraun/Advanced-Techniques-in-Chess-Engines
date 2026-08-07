#pragma once

#include "games/go/GoEncoding.hpp"
#include "games/go/GoTypes.hpp"
#include "util/py.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

enum class GoSymmetry : std::uint8_t {
    identity = 0,
    rotate_90 = 1,
    rotate_180 = 2,
    rotate_270 = 3,
    reflect = 4,
    reflect_rotate_90 = 5,
    reflect_rotate_180 = 6,
    reflect_rotate_270 = 7,
};

template <std::size_t BoardSize>
[[nodiscard]] typename BitBoard<BoardSize>::Point
transform_go_point(const typename BitBoard<BoardSize>::Point point, const GoSymmetry symmetry) {
    using Point = typename BitBoard<BoardSize>::Point;
    const std::uint8_t maximum = static_cast<std::uint8_t>(BoardSize - 1);
    switch (symmetry) {
    case GoSymmetry::identity:
        return point;
    case GoSymmetry::rotate_90:
        return Point{.x = static_cast<std::uint8_t>(maximum - point.y), .y = point.x};
    case GoSymmetry::rotate_180:
        return Point{.x = static_cast<std::uint8_t>(maximum - point.x),
                     .y = static_cast<std::uint8_t>(maximum - point.y)};
    case GoSymmetry::rotate_270:
        return Point{.x = point.y, .y = static_cast<std::uint8_t>(maximum - point.x)};
    case GoSymmetry::reflect:
        return Point{.x = static_cast<std::uint8_t>(maximum - point.x), .y = point.y};
    case GoSymmetry::reflect_rotate_90:
        return Point{.x = static_cast<std::uint8_t>(maximum - point.y),
                     .y = static_cast<std::uint8_t>(maximum - point.x)};
    case GoSymmetry::reflect_rotate_180:
        return Point{.x = point.x, .y = static_cast<std::uint8_t>(maximum - point.y)};
    case GoSymmetry::reflect_rotate_270:
        return Point{.x = point.y, .y = point.x};
    }
    throw std::invalid_argument("Unknown Go symmetry");
}

[[nodiscard]] inline GoSymmetry inverse_go_symmetry(const GoSymmetry symmetry) {
    switch (symmetry) {
    case GoSymmetry::identity:
        return GoSymmetry::identity;
    case GoSymmetry::rotate_90:
        return GoSymmetry::rotate_270;
    case GoSymmetry::rotate_180:
        return GoSymmetry::rotate_180;
    case GoSymmetry::rotate_270:
        return GoSymmetry::rotate_90;
    case GoSymmetry::reflect:
        return GoSymmetry::reflect;
    case GoSymmetry::reflect_rotate_90:
        return GoSymmetry::reflect_rotate_90;
    case GoSymmetry::reflect_rotate_180:
        return GoSymmetry::reflect_rotate_180;
    case GoSymmetry::reflect_rotate_270:
        return GoSymmetry::reflect_rotate_270;
    }
    throw std::invalid_argument("Unknown Go symmetry");
}

template <std::size_t BoardSize>
[[nodiscard]] GoAction<BoardSize> transform_go_action(const GoAction<BoardSize> action,
                                                      const GoSymmetry symmetry) {
    if (action.is_pass()) {
        return action;
    }
    return GoAction<BoardSize>(transform_go_point<BoardSize>(action.point(), symmetry));
}

template <std::size_t BoardSize, std::size_t HistoryLength>
[[nodiscard]] EncodedGoPosition<BoardSize, HistoryLength>
transform_go_encoding(const EncodedGoPosition<BoardSize, HistoryLength> &encoding,
                      const GoSymmetry symmetry) {
    EncodedGoPosition<BoardSize, HistoryLength> transformed{};
    for (const auto plane :
         range(EncodedGoPosition<BoardSize, HistoryLength>::binary_plane_count)) {
        for (const auto point : encoding.binary_planes[plane].set_bits()) {
            transformed.binary_planes[plane].set(transform_go_point<BoardSize>(point, symmetry));
        }
    }
    transformed.scalar_planes = encoding.scalar_planes;
    return transformed;
}
