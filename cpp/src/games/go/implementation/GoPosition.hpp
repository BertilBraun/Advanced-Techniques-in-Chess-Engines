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
        validateRules(rules);
    }

    [[nodiscard]] static GoPosition restore(const std::array<Board, HistoryLength> &history,
                                            const GoPlayer player,
                                            const std::optional<Point> koPoint,
                                            const int consecutivePasses, const int moveNumber,
                                            const GoRules rules) {
        validateRules(rules);
        for (const Board &historicBoard : history) {
            if (historicBoard.black.intersects(historicBoard.white)) {
                throw std::invalid_argument("Black and white Go stones overlap");
            }
        }
        if (consecutivePasses < 0 || consecutivePasses > 2 || consecutivePasses > moveNumber) {
            throw std::invalid_argument("Go consecutive pass count is invalid");
        }
        if (moveNumber < 0 || moveNumber > rules.maximum_moves) {
            throw std::invalid_argument("Go move number is outside the configured range");
        }
        if (koPoint.has_value() &&
            (history[0].black.test(*koPoint) || history[0].white.test(*koPoint))) {
            throw std::invalid_argument("Go ko point must be empty");
        }
        return GoPosition(history, player, koPoint, consecutivePasses, moveNumber, rules);
    }

    [[nodiscard]] const std::array<Board, HistoryLength> &history() const noexcept {
        return m_history;
    }
    [[nodiscard]] const Board &board() const noexcept { return m_history[0]; }
    [[nodiscard]] GoPlayer player() const noexcept { return m_player; }
    [[nodiscard]] std::optional<Point> koPoint() const noexcept { return m_koPoint; }
    [[nodiscard]] int consecutivePasses() const noexcept { return m_consecutivePasses; }
    [[nodiscard]] int moveNumber() const noexcept { return m_moveNumber; }
    [[nodiscard]] GoRules rules() const noexcept { return m_rules; }

    [[nodiscard]] bool isLegal(const GoAction<BoardSize> action) const {
        if (isTerminal()) {
            return false;
        }
        if (action.isPass()) {
            return true;
        }
        const Point point = action.point();
        if (m_koPoint == point || occupied().test(point)) {
            return false;
        }
        return placement(point).has_value();
    }

    [[nodiscard]] std::vector<GoAction<BoardSize>> legalActions() const {
        std::vector<GoAction<BoardSize>> actions;
        if (isTerminal()) {
            return actions;
        }
        actions.reserve(GoAction<BoardSize>::actionCount);
        for (const int id : range(GoAction<BoardSize>::passId)) {
            const GoAction<BoardSize> action(id);
            if (isLegal(action)) {
                actions.push_back(action);
            }
        }
        actions.push_back(GoAction<BoardSize>::pass());
        return actions;
    }

    [[nodiscard]] GoPosition child(const GoAction<BoardSize> action) const {
        if (!isLegal(action)) {
            throw std::invalid_argument("Illegal Go action");
        }
        std::array<Board, HistoryLength> nextHistory{};
        for (const auto historyOffset : range(HistoryLength - 1)) {
            const std::size_t destination = HistoryLength - historyOffset - 1;
            nextHistory[destination] = m_history[destination - 1];
        }
        std::optional<Point> nextKo;
        int nextPasses = m_consecutivePasses;
        if (action.isPass()) {
            nextHistory[0] = m_history[0];
            ++nextPasses;
        } else {
            const Point point = action.point();
            const Placement placed = *placement(point);
            nextHistory[0] = placed.board;
            nextKo = placed.koPoint;
            nextPasses = 0;
        }
        return GoPosition(nextHistory, opponent(m_player), nextKo, nextPasses, m_moveNumber + 1,
                          m_rules);
    }

    [[nodiscard]] GoTerminationReason terminationReason() const noexcept {
        if (m_consecutivePasses == 2) {
            return GoTerminationReason::two_passes;
        }
        if (m_moveNumber >= m_rules.maximum_moves) {
            return GoTerminationReason::maximum_moves;
        }
        return GoTerminationReason::ongoing;
    }

    [[nodiscard]] bool isTerminal() const noexcept {
        return terminationReason() != GoTerminationReason::ongoing;
    }

    [[nodiscard]] GoAreaScore areaScore() const {
        int blackArea = static_cast<int>(board().black.count());
        int whiteArea = static_cast<int>(board().white.count());
        BitBoard<BoardSize> unseen = ~occupied();
        Point origin{};
        while (unseen.popFirst(origin)) {
            BitBoard<BoardSize> region;
            BitBoard<BoardSize> frontier = BitBoard<BoardSize>::fromPoint(origin);
            bool touchesBlack = false;
            bool touchesWhite = false;
            Point point{};
            while (frontier.popFirst(point)) {
                if (region.test(point)) {
                    continue;
                }
                region.set(point);
                unseen.reset(point);
                forEachNeighbor(point, [&](const Point neighbor) {
                    if (board().black.test(neighbor)) {
                        touchesBlack = true;
                    } else if (board().white.test(neighbor)) {
                        touchesWhite = true;
                    } else if (!region.test(neighbor)) {
                        frontier.set(neighbor);
                    }
                });
            }
            if (touchesBlack && !touchesWhite) {
                blackArea += static_cast<int>(region.count());
            } else if (touchesWhite && !touchesBlack) {
                whiteArea += static_cast<int>(region.count());
            }
        }
        return GoAreaScore{
            .black_half_points = blackArea * 2,
            .white_half_points = whiteArea * 2 + m_rules.komi_half_points,
        };
    }

    [[nodiscard]] GoTerminalResult terminalResult() const {
        const GoTerminationReason reason = terminationReason();
        if (reason == GoTerminationReason::ongoing) {
            throw std::logic_error("Ongoing Go positions do not have a terminal result");
        }
        const GoAreaScore score = areaScore();
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
        for (const Board &historicBoard : m_history) {
            for (const std::uint64_t word : historicBoard.black.words()) {
                append(word);
            }
            for (const std::uint64_t word : historicBoard.white.words()) {
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
        std::optional<Point> koPoint;
    };

    GoPosition(const std::array<Board, HistoryLength> &history, const GoPlayer player,
               const std::optional<Point> koPoint, const int consecutivePasses,
               const int moveNumber, const GoRules rules)
        : m_history(history), m_player(player), m_koPoint(koPoint),
          m_consecutivePasses(consecutivePasses), m_moveNumber(moveNumber), m_rules(rules) {}

    static void validateRules(const GoRules rules) {
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
    static void forEachNeighbor(const Point point, Operation operation) {
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

    [[nodiscard]] static BitBoard<BoardSize> groupAt(const BitBoard<BoardSize> stones,
                                                     const Point origin) {
        BitBoard<BoardSize> group;
        BitBoard<BoardSize> frontier = BitBoard<BoardSize>::fromPoint(origin);
        Point point{};
        while (frontier.popFirst(point)) {
            if (group.test(point)) {
                continue;
            }
            group.set(point);
            forEachNeighbor(point, [&](const Point neighbor) {
                if (stones.test(neighbor) && !group.test(neighbor)) {
                    frontier.set(neighbor);
                }
            });
        }
        return group;
    }

    [[nodiscard]] static BitBoard<BoardSize> liberties(const BitBoard<BoardSize> group,
                                                       const BitBoard<BoardSize> occupiedPoints) {
        BitBoard<BoardSize> result;
        for (const Point point : group.setBits()) {
            forEachNeighbor(point, [&](const Point neighbor) {
                if (!occupiedPoints.test(neighbor)) {
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
        forEachNeighbor(point, [&](const Point neighbor) {
            if (!enemy.test(neighbor)) {
                return;
            }
            const BitBoard<BoardSize> group = groupAt(enemy, neighbor);
            if (liberties(group, own | enemy).none()) {
                captured |= group;
            }
        });
        enemy = enemy - captured;
        const BitBoard<BoardSize> placedGroup = groupAt(own, point);
        const BitBoard<BoardSize> placedLiberties = liberties(placedGroup, own | enemy);
        if (placedLiberties.none()) {
            return std::nullopt;
        }
        std::optional<Point> nextKo;
        if (captured.count() == 1 && placedGroup.count() == 1 && placedLiberties.count() == 1) {
            nextKo = BitBoard<BoardSize>::point(captured.firstSetIndex());
        }
        return Placement{.board = candidate, .koPoint = nextKo};
    }

    std::array<Board, HistoryLength> m_history;
    GoPlayer m_player;
    std::optional<Point> m_koPoint;
    int m_consecutivePasses;
    int m_moveNumber;
    GoRules m_rules;
};
