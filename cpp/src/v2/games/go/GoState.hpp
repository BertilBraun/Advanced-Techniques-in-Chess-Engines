#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace az::v2::games::go {

struct GoEncoding;

enum class Stone : std::int8_t { Empty = 0, Black = 1, White = 2 };
enum class Player : std::int8_t { Black = 1, White = 2 };
enum class TerminationReason : std::int8_t { Ongoing = 0, TwoPasses = 1, SafetyPlyCap = 2 };

inline constexpr std::int32_t maximum_history_length = 1024;

struct GoRules {
    std::int32_t board_size;
    std::int32_t komi_half_points;
    std::int32_t safety_ply_cap;
    std::int32_t history_length;

    [[nodiscard]] bool operator==(const GoRules &) const = default;
};

struct AreaScore {
    std::int64_t black_twice;
    std::int64_t white_twice;

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
    using action_type = std::int32_t;
    using player_type = Player;
    using termination_reason_type = TerminationReason;
    using terminal_result_type = TerminalResult;
    using encoding_type = GoEncoding;

    explicit GoState(GoRules rules);
    [[nodiscard]] static GoState restore(GoRules rules, std::vector<Stone> board,
                                         Player current_player, std::int32_t ply,
                                         std::int32_t consecutive_passes,
                                         std::vector<std::vector<Stone>> position_history);

    [[nodiscard]] const GoRules &rules() const;
    [[nodiscard]] std::int32_t board_size() const;
    [[nodiscard]] std::int32_t action_count() const;
    [[nodiscard]] std::int32_t pass_action() const;
    [[nodiscard]] Player current_player() const;
    [[nodiscard]] std::int32_t ply() const;
    [[nodiscard]] std::int32_t consecutive_passes() const;
    [[nodiscard]] const std::vector<Stone> &board() const;
    [[nodiscard]] const std::vector<std::vector<Stone>> &position_history() const;
    [[nodiscard]] bool is_legal(std::int32_t action) const;
    [[nodiscard]] std::vector<std::int32_t> legal_actions() const;
    void apply(std::int32_t action);
    [[nodiscard]] TerminationReason termination_reason() const;
    [[nodiscard]] bool is_terminal() const;
    [[nodiscard]] TerminalResult terminal_result() const;
    [[nodiscard]] AreaScore area_score() const;
    [[nodiscard]] GoEncoding canonical_encoding() const;
    [[nodiscard]] std::uint64_t state_hash() const;
    [[nodiscard]] bool operator==(const GoState &) const = default;

private:
    GoState(GoRules rules, std::vector<Stone> board, Player current_player, std::int32_t ply,
            std::int32_t consecutive_passes, TerminationReason termination_reason,
            std::vector<std::vector<Stone>> position_history);
    static void validate_rules(const GoRules &rules);

    [[nodiscard]] std::vector<Stone> board_after_placement(std::int32_t action) const;
    [[nodiscard]] std::vector<std::int32_t> group_at(const std::vector<Stone> &board,
                                                     std::int32_t origin) const;
    [[nodiscard]] bool group_has_liberty(const std::vector<Stone> &board,
                                         const std::vector<std::int32_t> &group) const;
    [[nodiscard]] std::vector<std::int32_t> neighbors(std::int32_t point) const;
    [[nodiscard]] bool repeats_position(const std::vector<Stone> &board) const;
    [[nodiscard]] static Stone stone_for(Player player);
    [[nodiscard]] static Player opponent(Player player);

    GoRules rules_;
    std::vector<Stone> board_;
    Player current_player_;
    std::int32_t ply_;
    std::int32_t consecutive_passes_;
    TerminationReason termination_reason_;
    std::vector<std::vector<Stone>> position_history_;
};

} // namespace az::v2::games::go
