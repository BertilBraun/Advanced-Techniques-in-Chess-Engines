#pragma once

#include "util/BitBoard.hpp"

#include <cassert>
#include <cstddef>
#include <stdexcept>

template <std::size_t BoardSize> struct GoAction {
    using Point = typename BitBoard<BoardSize>::Point;

    static constexpr int passId = static_cast<int>(BoardSize * BoardSize);
    static constexpr int actionCount = passId + 1;

    int id;

    explicit constexpr GoAction(const int actionId) : id(actionId) {
        if (actionId < 0 || actionId >= actionCount) {
            throw std::invalid_argument("Go action id is outside the action space");
        }
    }
    explicit constexpr GoAction(const Point point)
        : id(static_cast<int>(BitBoard<BoardSize>::index(point))) {}

    [[nodiscard]] static constexpr GoAction pass() { return GoAction(passId); }
    [[nodiscard]] constexpr bool isPass() const noexcept { return id == passId; }
    [[nodiscard]] constexpr Point point() const noexcept {
        assert(!isPass());
        return BitBoard<BoardSize>::point(static_cast<std::size_t>(id));
    }
    [[nodiscard]] bool operator==(const GoAction &) const noexcept = default;
};
