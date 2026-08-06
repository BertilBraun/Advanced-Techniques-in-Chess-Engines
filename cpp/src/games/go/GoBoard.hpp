#pragma once

#include "util/BitBoard.hpp"

#include <cstddef>

template <std::size_t BoardSize> struct GoBoard {
    BitBoard<BoardSize> black;
    BitBoard<BoardSize> white;

    [[nodiscard]] bool operator==(const GoBoard &) const noexcept = default;
};
