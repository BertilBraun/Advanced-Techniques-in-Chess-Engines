#pragma once

#include "games/go/implementation/GoAction.hpp"
#include "games/go/implementation/GoRules.hpp"
#include "util/py.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

template <std::size_t BoardSize> struct GoBoard {
    BitBoard<BoardSize> black;
    BitBoard<BoardSize> white;

    [[nodiscard]] bool operator==(const GoBoard &) const noexcept = default;
};

template <std::size_t BoardSize, std::size_t HistoryLength = 8> class GoPosition {
public:
    static_assert(HistoryLength >= 2);

    using Board = GoBoard<BoardSize>;
    using Point = typename BitBoard<BoardSize>::Point;

    explicit GoPosition(const GoRules rules)
        : m_history{}, m_player(GoPlayer::black), m_koPoint(std::nullopt), m_consecutivePasses(0),
          m_moveNumber(0), m_rules(rules) {
        validate_rules(rules);
    }

    [[nodiscard]] static GoPosition restore(const std::array<Board, HistoryLength> &history,
                                            const GoPlayer player,
                                            const std::optional<Point> ko_point,
                                            const int consecutive_passes, const int move_number,
                                            const GoRules rules) {
        validate_rules(rules);
        for (const Board &historic_board : history) {
            if (historic_board.black.intersects(historic_board.white)) {
                throw std::invalid_argument("Black and white Go stones overlap");
            }
        }
        if (consecutive_passes < 0 || consecutive_passes > 2 || consecutive_passes > move_number) {
            throw std::invalid_argument("Go consecutive pass count is invalid");
        }
        if (move_number < 0 || move_number > rules.maximum_moves) {
            throw std::invalid_argument("Go move number is outside the configured range");
        }
        if (ko_point.has_value() &&
            (history[0].black.test(*ko_point) || history[0].white.test(*ko_point))) {
            throw std::invalid_argument("Go ko point must be empty");
        }
        return GoPosition(history, player, ko_point, consecutive_passes, move_number, rules);
    }

    [[nodiscard]] const std::array<Board, HistoryLength> &history() const noexcept {
        return m_history;
    }
    [[nodiscard]] const Board &board() const noexcept { return m_history[0]; }
    [[nodiscard]] GoPlayer player() const noexcept { return m_player; }
    [[nodiscard]] std::optional<Point> ko_point() const noexcept { return m_koPoint; }
    [[nodiscard]] int consecutive_passes() const noexcept { return m_consecutivePasses; }
    [[nodiscard]] int move_number() const noexcept { return m_moveNumber; }
    [[nodiscard]] GoRules rules() const noexcept { return m_rules; }

    [[nodiscard]] bool is_legal(const GoAction<BoardSize> action) const {
        if (is_terminal()) {
            return false;
        }
        if (action.is_pass()) {
            return true;
        }
        const Point point = action.point();
        if (m_koPoint == point || occupied().test(point)) {
            return false;
        }
        return placement(point).has_value();
    }

    [[nodiscard]] std::vector<GoAction<BoardSize>> legal_actions() const {
        std::vector<GoAction<BoardSize>> actions;
        if (is_terminal()) {
            return actions;
        }
        actions.reserve(GoAction<BoardSize>::action_count);
        for (const int id : range(GoAction<BoardSize>::pass_id)) {
            const GoAction<BoardSize> action(id);
            if (is_legal(action)) {
                actions.push_back(action);
            }
        }
        actions.push_back(GoAction<BoardSize>::pass());
        return actions;
    }

    [[nodiscard]] GoPosition child(const GoAction<BoardSize> action) const {
        if (!is_legal(action)) {
            throw std::invalid_argument("Illegal Go action");
        }
        std::array<Board, HistoryLength> next_history{};
        for (const auto historyOffset : range(HistoryLength - 1)) {
            const std::size_t destination = HistoryLength - historyOffset - 1;
            next_history[destination] = m_history[destination - 1];
        }
        std::optional<Point> next_ko;
        int next_passes = m_consecutivePasses;
        if (action.is_pass()) {
            next_history[0] = m_history[0];
            ++next_passes;
        } else {
            const Point point = action.point();
            const Placement placed = *placement(point);
            next_history[0] = placed.board;
            next_ko = placed.ko_point;
            next_passes = 0;
        }
        return GoPosition(next_history, opponent(m_player), next_ko, next_passes, m_moveNumber + 1,
                          m_rules);
    }

    [[nodiscard]] GoTerminationReason termination_reason() const noexcept {
        if (m_consecutivePasses == 2) {
            return GoTerminationReason::two_passes;
        }
        if (m_moveNumber >= m_rules.maximum_moves) {
            return GoTerminationReason::maximum_moves;
        }
        return GoTerminationReason::ongoing;
    }

    [[nodiscard]] bool is_terminal() const noexcept {
        return termination_reason() != GoTerminationReason::ongoing;
    }

    [[nodiscard]] GoAreaScore area_score() const {
        int black_area = static_cast<int>(board().black.count());
        int white_area = static_cast<int>(board().white.count());
        BitBoard<BoardSize> unseen = ~occupied();
        Point origin{};
        while (unseen.pop_first(origin)) {
            BitBoard<BoardSize> region;
            BitBoard<BoardSize> frontier = BitBoard<BoardSize>::from_point(origin);
            bool touches_black = false;
            bool touches_white = false;
            Point point{};
            while (frontier.pop_first(point)) {
                if (region.test(point)) {
                    continue;
                }
                region.set(point);
                unseen.reset(point);
                for_each_neighbor(point, [&](const Point neighbor) {
                    if (board().black.test(neighbor)) {
                        touches_black = true;
                    } else if (board().white.test(neighbor)) {
                        touches_white = true;
                    } else if (!region.test(neighbor)) {
                        frontier.set(neighbor);
                    }
                });
            }
            if (touches_black && !touches_white) {
                black_area += static_cast<int>(region.count());
            } else if (touches_white && !touches_black) {
                white_area += static_cast<int>(region.count());
            }
        }
        return GoAreaScore{
            .black_half_points = black_area * 2,
            .white_half_points = white_area * 2 + m_rules.komi_half_points,
        };
    }

    [[nodiscard]] GoTerminalResult terminal_result() const {
        const GoTerminationReason reason = termination_reason();
        if (reason == GoTerminationReason::ongoing) {
            throw std::logic_error("Ongoing Go positions do not have a terminal result");
        }
        const GoAreaScore score = area_score();
        return GoTerminalResult{
            .reason = reason,
            .score = score,
            .winner = score.winner(),
        };
    }

    [[nodiscard]] std::uint64_t hash() const noexcept {
        std::uint64_t value = 14695981039346656037ULL;
        const auto append = [&value](const std::uint64_t item) {
            value ^= item;
            value *= 1099511628211ULL;
        };
        for (const Board &historic_board : m_history) {
            for (const std::uint64_t word : historic_board.black.words()) {
                append(word);
            }
            for (const std::uint64_t word : historic_board.white.words()) {
                append(word);
            }
        }
        append(static_cast<std::uint64_t>(m_player));
        append(m_koPoint.has_value() ? BitBoard<BoardSize>::index(*m_koPoint) + 1 : 0);
        append(static_cast<std::uint64_t>(m_consecutivePasses));
        append(static_cast<std::uint64_t>(m_moveNumber));
        append(static_cast<std::uint64_t>(static_cast<std::int64_t>(m_rules.komi_half_points)));
        append(static_cast<std::uint64_t>(m_rules.maximum_moves));
        return value;
    }

    [[nodiscard]] bool operator==(const GoPosition &) const noexcept = default;

private:
    struct Placement {
        Board board;
        std::optional<Point> ko_point;
    };

    GoPosition(const std::array<Board, HistoryLength> &history, const GoPlayer player,
               const std::optional<Point> ko_point, const int consecutive_passes,
               const int move_number, const GoRules rules)
        : m_history(history), m_player(player), m_koPoint(ko_point),
          m_consecutivePasses(consecutive_passes), m_moveNumber(move_number), m_rules(rules) {}

    static void validate_rules(const GoRules rules) {
        if (rules.maximum_moves < static_cast<int>(BoardSize * BoardSize)) {
            throw std::invalid_argument("Go maximum moves must be at least the board area");
        }
    }

    [[nodiscard]] static GoPlayer opponent(const GoPlayer player) noexcept {
        return player == GoPlayer::black ? GoPlayer::white : GoPlayer::black;
    }

    [[nodiscard]] BitBoard<BoardSize> occupied() const noexcept {
        return board().black | board().white;
    }

    template <typename Operation>
    static void for_each_neighbor(const Point point, Operation operation) {
        if (point.x > 0) {
            operation(Point{.x = static_cast<std::uint8_t>(point.x - 1), .y = point.y});
        }
        if (point.x + 1 < BoardSize) {
            operation(Point{.x = static_cast<std::uint8_t>(point.x + 1), .y = point.y});
        }
        if (point.y > 0) {
            operation(Point{.x = point.x, .y = static_cast<std::uint8_t>(point.y - 1)});
        }
        if (point.y + 1 < BoardSize) {
            operation(Point{.x = point.x, .y = static_cast<std::uint8_t>(point.y + 1)});
        }
    }

    [[nodiscard]] static BitBoard<BoardSize> group_at(const BitBoard<BoardSize> stones,
                                                      const Point origin) {
        BitBoard<BoardSize> group;
        BitBoard<BoardSize> frontier = BitBoard<BoardSize>::from_point(origin);
        Point point{};
        while (frontier.pop_first(point)) {
            if (group.test(point)) {
                continue;
            }
            group.set(point);
            for_each_neighbor(point, [&](const Point neighbor) {
                if (stones.test(neighbor) && !group.test(neighbor)) {
                    frontier.set(neighbor);
                }
            });
        }
        return group;
    }

    [[nodiscard]] static BitBoard<BoardSize> liberties(const BitBoard<BoardSize> group,
                                                       const BitBoard<BoardSize> occupied_points) {
        BitBoard<BoardSize> result;
        for (const Point point : group.set_bits()) {
            for_each_neighbor(point, [&](const Point neighbor) {
                if (!occupied_points.test(neighbor)) {
                    result.set(neighbor);
                }
            });
        }
        return result;
    }

    [[nodiscard]] std::optional<Placement> placement(const Point point) const {
        Board candidate = board();
        BitBoard<BoardSize> &own = m_player == GoPlayer::black ? candidate.black : candidate.white;
        BitBoard<BoardSize> &enemy =
            m_player == GoPlayer::black ? candidate.white : candidate.black;
        own.set(point);
        BitBoard<BoardSize> captured;
        for_each_neighbor(point, [&](const Point neighbor) {
            if (!enemy.test(neighbor)) {
                return;
            }
            const BitBoard<BoardSize> group = group_at(enemy, neighbor);
            if (liberties(group, own | enemy).none()) {
                captured |= group;
            }
        });
        enemy = enemy - captured;
        const BitBoard<BoardSize> placed_group = group_at(own, point);
        const BitBoard<BoardSize> placed_liberties = liberties(placed_group, own | enemy);
        if (placed_liberties.none()) {
            return std::nullopt;
        }
        std::optional<Point> next_ko;
        if (captured.count() == 1 && placed_group.count() == 1 && placed_liberties.count() == 1) {
            next_ko = BitBoard<BoardSize>::point(captured.first_set_index());
        }
        return Placement{.board = candidate, .ko_point = next_ko};
    }

    std::array<Board, HistoryLength> m_history;
    GoPlayer m_player;
    std::optional<Point> m_koPoint;
    int m_consecutivePasses;
    int m_moveNumber;
    GoRules m_rules;
};
