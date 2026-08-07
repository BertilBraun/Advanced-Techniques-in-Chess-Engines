#pragma once

#include <cstddef>
#include <stdexcept>

template <std::size_t BoardSize> struct GoAction {
    static constexpr int pass_id = static_cast<int>(BoardSize * BoardSize);
    static constexpr int action_count = pass_id + 1;

    int id;

    explicit constexpr GoAction(const int action_id) : id(action_id) {
        if (action_id < 0 || action_id >= action_count) {
            throw std::invalid_argument("Go action id is outside the action space");
        }
    }

    [[nodiscard]] static constexpr GoAction pass() { return GoAction(pass_id); }
    [[nodiscard]] constexpr bool is_pass() const noexcept { return id == pass_id; }
    [[nodiscard]] bool operator==(const GoAction &) const noexcept = default;
};
