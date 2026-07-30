#pragma once

#include "common.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace az::v2::games::go {

struct GoEncoding;

enum class Stone : int8 { Empty = 0, Black = 1, White = 2 };
enum class Player : int8 { Black = 1, White = 2 };
enum class TerminationReason : int8 { Ongoing = 0, TwoPasses = 1, SafetyPlyCap = 2 };

inline constexpr int32 MAXIMUM_HISTORY_LENGTH = 1024;

struct GoRules {
    int32 boardSize;
    int32 komiHalfPoints;
    int32 safetyPlyCap;
    int32 historyLength;

    [[nodiscard]] bool operator==(const GoRules &) const = default;
};

struct AreaScore {
    int64 blackTwice;
    int64 whiteTwice;

    [[nodiscard]] std::optional<Player> winner() const;
    [[nodiscard]] bool operator==(const AreaScore &) const = default;
};

struct TerminalResult {
    TerminationReason reason;
    std::optional<AreaScore> score;
    std::optional<Player> winner;

    [[nodiscard]] bool operator==(const TerminalResult &) const = default;
};

class GoState {
public:
    using action_type = int32;
    using player_type = Player;
    using termination_reason_type = TerminationReason;
    using terminal_result_type = TerminalResult;
    using encoding_type = GoEncoding;

    explicit GoState(GoRules rules);
    [[nodiscard]] static GoState restore(GoRules rules, std::vector<Stone> board,
                                         Player currentPlayer, int32 ply, int32 consecutivePasses,
                                         std::vector<std::vector<Stone>> positionHistory);

    [[nodiscard]] const GoRules &rules() const;
    [[nodiscard]] int32 boardSize() const;
    [[nodiscard]] int32 actionCount() const;
    [[nodiscard]] int32 passAction() const;
    [[nodiscard]] Player currentPlayer() const;
    [[nodiscard]] int32 ply() const;
    [[nodiscard]] int32 consecutivePasses() const;
    [[nodiscard]] const std::vector<Stone> &board() const;
    [[nodiscard]] const std::vector<std::vector<Stone>> &positionHistory() const;
    [[nodiscard]] bool isLegal(int32 action) const;
    [[nodiscard]] std::vector<int32> legalActions() const;
    void apply(int32 action);
    [[nodiscard]] TerminationReason terminationReason() const;
    [[nodiscard]] bool isTerminal() const;
    [[nodiscard]] TerminalResult terminalResult() const;
    [[nodiscard]] AreaScore areaScore() const;
    [[nodiscard]] GoEncoding canonicalEncoding() const;
    [[nodiscard]] uint64 stateHash() const;
    [[nodiscard]] bool operator==(const GoState &) const = default;

private:
    GoState(GoRules rules, std::vector<Stone> board, Player currentPlayer, int32 ply,
            int32 consecutivePasses, TerminationReason terminationReason,
            std::vector<std::vector<Stone>> positionHistory);
    static void validateRules(const GoRules &rules);

    [[nodiscard]] std::vector<Stone> boardAfterPlacement(int32 action) const;
    [[nodiscard]] std::vector<int32> groupAt(const std::vector<Stone> &board, int32 origin) const;
    [[nodiscard]] bool groupHasLiberty(const std::vector<Stone> &board,
                                       const std::vector<int32> &group) const;
    [[nodiscard]] std::vector<int32> neighbors(int32 point) const;
    [[nodiscard]] bool repeatsPosition(const std::vector<Stone> &board) const;
    [[nodiscard]] static Stone stoneFor(Player player);
    [[nodiscard]] static Player opponent(Player player);

    GoRules _rules;
    std::vector<Stone> _board;
    Player _currentPlayer;
    int32 _ply;
    int32 _consecutivePasses;
    TerminationReason _terminationReason;
    std::vector<std::vector<Stone>> _positionHistory;
};

} // namespace az::v2::games::go
