#include "games/go/GoState.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace az::v2::games::go {
namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

void hash_byte(std::uint64_t &hash, std::uint8_t value) {
    hash ^= value;
    hash *= fnv_prime;
}

void hash_int(std::uint64_t &hash, std::int32_t value) {
    const auto unsigned_value = static_cast<std::uint32_t>(value);
    for (std::int32_t shift = 0; shift < 32; shift += 8) {
        hash_byte(hash, static_cast<std::uint8_t>((unsigned_value >> shift) & 0xffU));
    }
}

} // namespace

std::optional<Player> AreaScore::winner() const {
    if (black_twice > white_twice) {
        return Player::Black;
    }
    if (white_twice > black_twice) {
        return Player::White;
    }
    return std::nullopt;
}

GoState::GoState(GoRules rules)
    : rules_(rules), board_(), current_player_(Player::Black), ply_(0), consecutive_passes_(0),
      termination_reason_(TerminationReason::Ongoing), position_history_() {
    validate_rules(rules);
    const auto point_count =
        static_cast<std::size_t>(rules.board_size) * static_cast<std::size_t>(rules.board_size);
    board_.assign(point_count, Stone::Empty);
    position_history_.push_back(board_);
}

GoState::GoState(GoRules rules, std::vector<Stone> board, Player current_player, std::int32_t ply,
                 std::int32_t consecutive_passes, TerminationReason termination_reason,
                 std::vector<std::vector<Stone>> position_history)
    : rules_(rules), board_(std::move(board)), current_player_(current_player), ply_(ply),
      consecutive_passes_(consecutive_passes), termination_reason_(termination_reason),
      position_history_(std::move(position_history)) {}

void GoState::validate_rules(const GoRules &rules) {
    if (rules.board_size != 7 && rules.board_size != 9) {
        throw std::invalid_argument("Go board size must be 7 or 9");
    }
    const std::int32_t minimum_ply_cap = rules.board_size * rules.board_size;
    if (rules.safety_ply_cap < minimum_ply_cap) {
        throw std::invalid_argument("Go safety ply cap must be at least the board area");
    }
    if (rules.history_length < 1) {
        throw std::invalid_argument("Go history length must be positive");
    }
    if (rules.history_length > maximum_history_length) {
        throw std::invalid_argument("Go history length exceeds the supported maximum");
    }
}

GoState GoState::restore(GoRules rules, std::vector<Stone> board, Player current_player,
                         std::int32_t ply, std::int32_t consecutive_passes,
                         std::vector<std::vector<Stone>> position_history) {
    validate_rules(rules);
    const auto point_count =
        static_cast<std::size_t>(rules.board_size) * static_cast<std::size_t>(rules.board_size);
    if (board.size() != point_count || position_history.empty() ||
        position_history.back() != board) {
        throw std::invalid_argument("Restored Go board history is inconsistent");
    }
    if (std::any_of(position_history.begin(), position_history.end(),
                    [point_count](const std::vector<Stone> &historic_board) {
                        return historic_board.size() != point_count ||
                               std::any_of(
                                   historic_board.begin(), historic_board.end(), [](Stone stone) {
                                       return stone != Stone::Empty && stone != Stone::Black &&
                                              stone != Stone::White;
                                   });
                    })) {
        throw std::invalid_argument("Restored Go board history contains invalid data");
    }
    if (ply < 0 || static_cast<std::size_t>(ply) + 1U != position_history.size()) {
        throw std::invalid_argument("Restored Go ply does not match its position history");
    }
    if (ply > rules.safety_ply_cap) {
        throw std::invalid_argument("Restored Go ply exceeds the safety ply cap");
    }
    const Player expected_player = ply % 2 == 0 ? Player::Black : Player::White;
    if (current_player != expected_player) {
        throw std::invalid_argument("Restored Go player does not match its ply");
    }
    if (consecutive_passes < 0 || consecutive_passes > 2 || consecutive_passes > ply) {
        throw std::invalid_argument("Restored Go consecutive pass count is invalid");
    }
    if (ply > 0) {
        const auto &latest = position_history[position_history.size() - 1U];
        const auto &previous = position_history[position_history.size() - 2U];
        if ((consecutive_passes == 0) != (latest != previous)) {
            throw std::invalid_argument("Restored Go pass count does not match its history");
        }
        if (consecutive_passes == 2) {
            if (position_history.size() < 3U ||
                previous != position_history[position_history.size() - 3U]) {
                throw std::invalid_argument("Restored Go double pass does not match its history");
            }
        } else if (consecutive_passes == 1 && position_history.size() >= 3U &&
                   previous == position_history[position_history.size() - 3U]) {
            throw std::invalid_argument(
                "Restored Go single pass follows another pass in its history");
        }
    }
    TerminationReason reason = TerminationReason::Ongoing;
    if (consecutive_passes == 2) {
        reason = TerminationReason::TwoPasses;
    } else if (ply >= rules.safety_ply_cap) {
        reason = TerminationReason::SafetyPlyCap;
    }
    return GoState(rules, std::move(board), current_player, ply, consecutive_passes, reason,
                   std::move(position_history));
}

const GoRules &GoState::rules() const { return rules_; }
std::int32_t GoState::board_size() const { return rules_.board_size; }
std::int32_t GoState::action_count() const { return rules_.board_size * rules_.board_size + 1; }
std::int32_t GoState::pass_action() const { return action_count() - 1; }
Player GoState::current_player() const { return current_player_; }
std::int32_t GoState::ply() const { return ply_; }
std::int32_t GoState::consecutive_passes() const { return consecutive_passes_; }
const std::vector<Stone> &GoState::board() const { return board_; }
const std::vector<std::vector<Stone>> &GoState::position_history() const {
    return position_history_;
}

std::vector<std::int32_t> GoState::neighbors(std::int32_t point) const {
    const std::int32_t row = point / rules_.board_size;
    const std::int32_t column = point % rules_.board_size;
    std::vector<std::int32_t> result;
    result.reserve(4);
    if (row > 0) {
        result.push_back(point - rules_.board_size);
    }
    if (column > 0) {
        result.push_back(point - 1);
    }
    if (column + 1 < rules_.board_size) {
        result.push_back(point + 1);
    }
    if (row + 1 < rules_.board_size) {
        result.push_back(point + rules_.board_size);
    }
    return result;
}

std::vector<std::int32_t> GoState::group_at(const std::vector<Stone> &board,
                                            std::int32_t origin) const {
    const Stone color = board[static_cast<std::size_t>(origin)];
    std::vector<std::int32_t> group;
    std::vector<std::int32_t> pending{origin};
    std::vector<bool> seen(board.size(), false);
    seen[static_cast<std::size_t>(origin)] = true;
    while (!pending.empty()) {
        const std::int32_t point = pending.back();
        pending.pop_back();
        group.push_back(point);
        for (const std::int32_t neighbor : neighbors(point)) {
            const auto index = static_cast<std::size_t>(neighbor);
            if (!seen[index] && board[index] == color) {
                seen[index] = true;
                pending.push_back(neighbor);
            }
        }
    }
    return group;
}

bool GoState::group_has_liberty(const std::vector<Stone> &board,
                                const std::vector<std::int32_t> &group) const {
    for (const std::int32_t point : group) {
        for (const std::int32_t neighbor : neighbors(point)) {
            if (board[static_cast<std::size_t>(neighbor)] == Stone::Empty) {
                return true;
            }
        }
    }
    return false;
}

std::vector<Stone> GoState::board_after_placement(std::int32_t action) const {
    std::vector<Stone> candidate = board_;
    const Stone own_stone = stone_for(current_player_);
    const Stone opponent_stone = stone_for(opponent(current_player_));
    candidate[static_cast<std::size_t>(action)] = own_stone;

    for (const std::int32_t neighbor : neighbors(action)) {
        if (candidate[static_cast<std::size_t>(neighbor)] != opponent_stone) {
            continue;
        }
        const std::vector<std::int32_t> group = group_at(candidate, neighbor);
        if (!group_has_liberty(candidate, group)) {
            for (const std::int32_t point : group) {
                candidate[static_cast<std::size_t>(point)] = Stone::Empty;
            }
        }
    }
    if (!group_has_liberty(candidate, group_at(candidate, action))) {
        throw std::invalid_argument("Go placement is suicide");
    }
    return candidate;
}

bool GoState::repeats_position(const std::vector<Stone> &board) const {
    return std::find(position_history_.begin(), position_history_.end(), board) !=
           position_history_.end();
}

bool GoState::is_legal(std::int32_t action) const {
    if (is_terminal() || action < 0 || action >= action_count()) {
        return false;
    }
    if (action == pass_action()) {
        return true;
    }
    if (board_[static_cast<std::size_t>(action)] != Stone::Empty) {
        return false;
    }
    try {
        return !repeats_position(board_after_placement(action));
    } catch (const std::invalid_argument &) {
        return false;
    }
}

std::vector<std::int32_t> GoState::legal_actions() const {
    std::vector<std::int32_t> result;
    if (is_terminal()) {
        return result;
    }
    for (std::int32_t action = 0; action < action_count(); ++action) {
        if (is_legal(action)) {
            result.push_back(action);
        }
    }
    return result;
}

void GoState::apply(std::int32_t action) {
    if (!is_legal(action)) {
        throw std::invalid_argument("Illegal Go action");
    }
    if (action == pass_action()) {
        ++consecutive_passes_;
    } else {
        board_ = board_after_placement(action);
        consecutive_passes_ = 0;
    }
    ++ply_;
    current_player_ = opponent(current_player_);
    position_history_.push_back(board_);
    if (consecutive_passes_ == 2) {
        termination_reason_ = TerminationReason::TwoPasses;
    } else if (ply_ >= rules_.safety_ply_cap) {
        termination_reason_ = TerminationReason::SafetyPlyCap;
    }
}

TerminationReason GoState::termination_reason() const { return termination_reason_; }
bool GoState::is_terminal() const { return termination_reason_ != TerminationReason::Ongoing; }

AreaScore GoState::area_score() const {
    std::int64_t black_area = 0;
    std::int64_t white_area = 0;
    std::vector<bool> seen(board_.size(), false);
    for (std::int32_t point = 0; point < pass_action(); ++point) {
        const auto index = static_cast<std::size_t>(point);
        if (board_[index] == Stone::Black) {
            ++black_area;
            continue;
        }
        if (board_[index] == Stone::White) {
            ++white_area;
            continue;
        }
        if (seen[index]) {
            continue;
        }
        std::vector<std::int32_t> region;
        std::vector<std::int32_t> pending{point};
        seen[index] = true;
        bool touches_black = false;
        bool touches_white = false;
        while (!pending.empty()) {
            const std::int32_t empty_point = pending.back();
            pending.pop_back();
            region.push_back(empty_point);
            for (const std::int32_t neighbor : neighbors(empty_point)) {
                const Stone stone = board_[static_cast<std::size_t>(neighbor)];
                if (stone == Stone::Black) {
                    touches_black = true;
                } else if (stone == Stone::White) {
                    touches_white = true;
                } else if (!seen[static_cast<std::size_t>(neighbor)]) {
                    seen[static_cast<std::size_t>(neighbor)] = true;
                    pending.push_back(neighbor);
                }
            }
        }
        if (touches_black && !touches_white) {
            black_area += static_cast<std::int64_t>(region.size());
        } else if (touches_white && !touches_black) {
            white_area += static_cast<std::int64_t>(region.size());
        }
    }
    return AreaScore{
        .black_twice = black_area * 2,
        .white_twice = white_area * 2 + static_cast<std::int64_t>(rules_.komi_half_points),
    };
}

TerminalResult GoState::terminal_result() const {
    if (termination_reason_ == TerminationReason::Ongoing) {
        return TerminalResult{
            .reason = termination_reason_, .score = std::nullopt, .winner = std::nullopt};
    }
    if (termination_reason_ == TerminationReason::SafetyPlyCap) {
        return TerminalResult{
            .reason = termination_reason_, .score = std::nullopt, .winner = std::nullopt};
    }
    const AreaScore score = area_score();
    return TerminalResult{.reason = termination_reason_, .score = score, .winner = score.winner()};
}

std::uint64_t GoState::state_hash() const {
    std::uint64_t hash = fnv_offset;
    hash_int(hash, rules_.board_size);
    hash_int(hash, rules_.komi_half_points);
    hash_int(hash, rules_.safety_ply_cap);
    hash_int(hash, rules_.history_length);
    hash_byte(hash, static_cast<std::uint8_t>(current_player_));
    hash_int(hash, ply_);
    hash_int(hash, consecutive_passes_);
    hash_byte(hash, static_cast<std::uint8_t>(termination_reason_));
    for (const auto &historic_board : position_history_) {
        for (const Stone stone : historic_board) {
            hash_byte(hash, static_cast<std::uint8_t>(stone));
        }
        hash_byte(hash, std::numeric_limits<std::uint8_t>::max());
    }
    return hash;
}

Stone GoState::stone_for(Player player) { return static_cast<Stone>(player); }
Player GoState::opponent(Player player) {
    return player == Player::Black ? Player::White : Player::Black;
}

} // namespace az::v2::games::go
