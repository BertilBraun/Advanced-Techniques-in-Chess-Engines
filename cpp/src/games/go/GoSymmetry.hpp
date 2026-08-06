#pragma once

#include "games/go/GoAction.hpp"
#include "games/go/GoEncoding.hpp"
#include "games/go/GoSymmetryTypes.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

template <std::size_t BoardSize>
[[nodiscard]] typename BitBoard<BoardSize>::Point
transform_go_point(const typename BitBoard<BoardSize>::Point point, const GoSymmetry symmetry) {
    using Point = typename BitBoard<BoardSize>::Point;
    const std::uint8_t maximum = static_cast<std::uint8_t>(BoardSize - 1);
    switch (symmetry) {
    case GoSymmetry::identity:
        return point;
    case GoSymmetry::rotate_90:
        return Point{static_cast<std::uint8_t>(maximum - point.y), point.x};
    case GoSymmetry::rotate_180:
        return Point{static_cast<std::uint8_t>(maximum - point.x),
                     static_cast<std::uint8_t>(maximum - point.y)};
    case GoSymmetry::rotate_270:
        return Point{point.y, static_cast<std::uint8_t>(maximum - point.x)};
    case GoSymmetry::reflect:
        return Point{static_cast<std::uint8_t>(maximum - point.x), point.y};
    case GoSymmetry::reflect_rotate_90:
        return Point{static_cast<std::uint8_t>(maximum - point.y),
                     static_cast<std::uint8_t>(maximum - point.x)};
    case GoSymmetry::reflect_rotate_180:
        return Point{point.x, static_cast<std::uint8_t>(maximum - point.y)};
    case GoSymmetry::reflect_rotate_270:
        return Point{point.y, point.x};
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
    return GoAction<BoardSize>(
        static_cast<int>(BitBoard<BoardSize>::index(transform_go_point<BoardSize>(
            BitBoard<BoardSize>::point(static_cast<std::size_t>(action.id)), symmetry))));
}

template <std::size_t BoardSize, std::size_t HistoryLength>
[[nodiscard]] EncodedGoPosition<BoardSize, HistoryLength>
transform_go_encoding(const EncodedGoPosition<BoardSize, HistoryLength> &encoding,
                      const GoSymmetry symmetry) {
    EncodedGoPosition<BoardSize, HistoryLength> transformed{};
    for (std::size_t plane = 0;
         plane < EncodedGoPosition<BoardSize, HistoryLength>::binary_plane_count; ++plane) {
        for (const auto point : encoding.binary_planes[plane].set_bits()) {
            transformed.binary_planes[plane].set(transform_go_point<BoardSize>(point, symmetry));
        }
    }
    transformed.scalar_planes = encoding.scalar_planes;
    return transformed;
}
