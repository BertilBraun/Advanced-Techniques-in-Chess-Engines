#include "games/go/GoEncoding.hpp"

#include <stdexcept>

namespace az::v2::games::go {

std::int8_t GoEncoding::at(std::int32_t plane, std::int32_t row, std::int32_t column) const {
    if (board_size != 7 && board_size != 9) {
        throw std::invalid_argument("Go board size must be 7 or 9");
    }
    if (plane < 0 || plane >= planes || row < 0 || row >= board_size || column < 0 ||
        column >= board_size) {
        throw std::out_of_range("Go encoding coordinate out of range");
    }
    const auto plane_size =
        static_cast<std::size_t>(board_size) * static_cast<std::size_t>(board_size);
    if (values.size() != static_cast<std::size_t>(planes) * plane_size) {
        throw std::invalid_argument("Go encoding shape is inconsistent");
    }
    const auto index = static_cast<std::size_t>(plane) * plane_size +
                       static_cast<std::size_t>(row) * static_cast<std::size_t>(board_size) +
                       static_cast<std::size_t>(column);
    return values[index];
}

GoEncoding canonical_encoding(const GoState &state) {
    const std::int32_t history_length = state.rules().history_length;
    const std::int32_t board_size = state.board_size();
    const std::int32_t plane_count = history_length * 2 + 1;
    const auto plane_size =
        static_cast<std::size_t>(board_size) * static_cast<std::size_t>(board_size);
    GoEncoding encoding{
        .planes = plane_count,
        .board_size = board_size,
        .values = std::vector<std::int8_t>(static_cast<std::size_t>(plane_count) * plane_size, 0),
    };
    const Stone own = state.current_player() == Player::Black ? Stone::Black : Stone::White;
    const Stone opponent = own == Stone::Black ? Stone::White : Stone::Black;
    const auto &history = state.position_history();
    for (std::int32_t offset = 0; offset < history_length; ++offset) {
        if (static_cast<std::size_t>(offset) >= history.size()) {
            break;
        }
        const auto &board = history[history.size() - 1U - static_cast<std::size_t>(offset)];
        for (std::size_t point = 0; point < plane_size; ++point) {
            const Stone stone = board[point];
            if (stone == own) {
                const auto plane = static_cast<std::size_t>(offset) * 2U;
                encoding.values[plane * plane_size + point] = 1;
            } else if (stone == opponent) {
                const auto plane = static_cast<std::size_t>(offset) * 2U + 1U;
                encoding.values[plane * plane_size + point] = 1;
            }
        }
    }
    if (state.current_player() == Player::Black) {
        const auto color_plane = static_cast<std::size_t>(plane_count - 1);
        for (std::size_t point = 0; point < plane_size; ++point) {
            encoding.values[color_plane * plane_size + point] = 1;
        }
    }
    return encoding;
}

GoEncoding GoState::canonical_encoding() const { return go::canonical_encoding(*this); }

} // namespace az::v2::games::go
