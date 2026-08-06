#pragma once

#include <cstdint>

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
