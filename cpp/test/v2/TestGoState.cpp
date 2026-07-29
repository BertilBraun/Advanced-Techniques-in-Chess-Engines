#include "games/game_concepts.hpp"
#include "games/go/GoEncoding.hpp"
#include "games/go/GoState.hpp"
#include "games/go/GoSymmetry.hpp"

#include <algorithm>
#include <array>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <limits>
#include <stdexcept>
#include <vector>

using az::v2::games::go::GoRules;
using az::v2::games::go::GoState;
using az::v2::games::go::GoSymmetryOperations;
using az::v2::games::go::Player;
using az::v2::games::go::Stone;
using az::v2::games::go::Symmetry;
using az::v2::games::go::TerminationReason;

static_assert(az::v2::games::GameState<GoState>);
static_assert(az::v2::games::GameSymmetry<GoSymmetryOperations>);

namespace {

GoRules rules(std::int32_t size = 7, std::int32_t cap = 200) {
    return GoRules{
        .board_size = size,
        .komi_half_points = 15,
        .safety_ply_cap = cap,
        .history_length = 4,
    };
}

std::vector<Stone> board_with(std::initializer_list<std::pair<std::int32_t, Stone>> stones) {
    std::vector<Stone> board(49, Stone::Empty);
    for (const auto &[point, stone] : stones) {
        board[static_cast<std::size_t>(point)] = stone;
    }
    return board;
}

void test_configuration_and_passes() {
    for (const std::int32_t size : {7, 9}) {
        GoState state(rules(size));
        assert(state.action_count() == size * size + 1);
        assert(state.pass_action() == size * size);
        assert(state.legal_actions().size() == static_cast<std::size_t>(size * size + 1));
        state.apply(state.pass_action());
        assert(!state.is_terminal());
        assert(state.is_legal(state.pass_action()));
        state.apply(state.pass_action());
        assert(state.is_terminal());
        assert(state.termination_reason() == TerminationReason::TwoPasses);
        assert(state.legal_actions().empty());
        assert(state.terminal_result().score.has_value());
        assert(state.terminal_result().winner == Player::White);
    }
}

void test_capture_and_suicide() {
    const auto capture_board =
        board_with({{8, Stone::White}, {1, Stone::Black}, {7, Stone::Black}, {9, Stone::Black}});
    GoState capture =
        GoState::restore(rules(), capture_board, Player::Black, 0, 0, {capture_board});
    capture.apply(15);
    assert(capture.board()[8] == Stone::Empty);

    const auto multi_capture_board = board_with({{8, Stone::White},
                                                 {9, Stone::White},
                                                 {1, Stone::Black},
                                                 {2, Stone::Black},
                                                 {7, Stone::Black},
                                                 {10, Stone::Black},
                                                 {15, Stone::Black}});
    GoState multi_capture =
        GoState::restore(rules(), multi_capture_board, Player::Black, 0, 0, {multi_capture_board});
    multi_capture.apply(16);
    assert(multi_capture.board()[8] == Stone::Empty);
    assert(multi_capture.board()[9] == Stone::Empty);

    const auto suicide_board =
        board_with({{1, Stone::Black}, {2, Stone::White}, {7, Stone::White}, {8, Stone::White}});
    const GoState suicide =
        GoState::restore(rules(), suicide_board, Player::Black, 0, 0, {suicide_board});
    assert(!suicide.is_legal(0));
}

void test_positional_superko() {
    const auto before = board_with({{8, Stone::White},
                                    {14, Stone::White},
                                    {16, Stone::White},
                                    {22, Stone::White},
                                    {1, Stone::Black},
                                    {7, Stone::Black},
                                    {9, Stone::Black}});
    GoState ko = GoState::restore(rules(), before, Player::Black, 0, 0, {before});
    ko.apply(15);
    assert(!ko.is_legal(8));

    const auto repeated = board_with({{14, Stone::White},
                                      {16, Stone::White},
                                      {22, Stone::White},
                                      {1, Stone::Black},
                                      {7, Stone::Black},
                                      {9, Stone::Black},
                                      {15, Stone::Black}});
    const auto middle = board_with({{30, Stone::White}});
    const GoState longer_cycle =
        GoState::restore(rules(), before, Player::Black, 2, 0, {repeated, middle, before});
    assert(!longer_cycle.is_legal(15));
}

void test_copy_hash_and_cap() {
    GoState state(rules());
    assert(state.state_hash() == 6493982775080899741ULL);
    state.apply(0);
    GoState copy = state;
    assert(copy == state);
    assert(copy.state_hash() == state.state_hash());
    copy.apply(1);
    assert(!(copy == state));
    assert(copy.state_hash() != state.state_hash());
    GoState long_game(rules());
    for (std::int32_t ply = 0; ply < 48; ++ply) {
        const auto legal = long_game.legal_actions();
        const auto placement =
            std::find_if(legal.begin(), legal.end(), [&long_game](std::int32_t action) {
                return action != long_game.pass_action();
            });
        assert(placement != legal.end());
        long_game.apply(*placement);
    }
    GoState capped = GoState::restore(rules(7, 49), long_game.board(), long_game.current_player(),
                                      long_game.ply(), long_game.consecutive_passes(),
                                      long_game.position_history());
    capped.apply(capped.pass_action());
    assert(capped.termination_reason() == TerminationReason::SafetyPlyCap);
    assert(!capped.terminal_result().score.has_value());
    assert(!capped.terminal_result().winner.has_value());
}

void test_restored_pass_invariants() {
    const auto empty = board_with({});
    const auto placed = board_with({{0, Stone::Black}});
    const GoState no_pass = GoState::restore(rules(), placed, Player::White, 1, 0, {empty, placed});
    const GoState one_pass =
        GoState::restore(rules(), placed, Player::Black, 2, 1, {empty, placed, placed});
    const GoState two_passes =
        GoState::restore(rules(), placed, Player::White, 3, 2, {empty, placed, placed, placed});
    assert(no_pass.consecutive_passes() == 0);
    assert(one_pass.consecutive_passes() == 1);
    assert(two_passes.termination_reason() == TerminationReason::TwoPasses);

    bool rejected_zero_with_duplicate = false;
    try {
        (void) GoState::restore(rules(), empty, Player::White, 1, 0, {empty, empty});
    } catch (const std::invalid_argument &) {
        rejected_zero_with_duplicate = true;
    }
    assert(rejected_zero_with_duplicate);

    bool rejected_single_after_pass = false;
    try {
        (void) GoState::restore(rules(), placed, Player::White, 3, 1,
                                {empty, placed, placed, placed});
    } catch (const std::invalid_argument &) {
        rejected_single_after_pass = true;
    }
    assert(rejected_single_after_pass);

    bool rejected_incomplete_double_pass = false;
    try {
        (void) GoState::restore(rules(), placed, Player::Black, 2, 2, {empty, placed, placed});
    } catch (const std::invalid_argument &) {
        rejected_incomplete_double_pass = true;
    }
    assert(rejected_incomplete_double_pass);
}

void test_area_scoring_with_integer_komi() {
    const auto neutral_board = board_with({{0, Stone::Black}, {48, Stone::White}});
    const GoState state =
        GoState::restore(rules(), neutral_board, Player::Black, 0, 0, {neutral_board});
    const auto score = state.area_score();
    assert(score.black_twice == 2);
    assert(score.white_twice == 17);
    assert(score.winner() == Player::White);

    const auto surrounded_corner = board_with({{1, Stone::Black}, {7, Stone::Black}});
    const GoState territory =
        GoState::restore(rules(), surrounded_corner, Player::Black, 0, 0, {surrounded_corner});
    const auto territory_score = territory.area_score();
    assert(territory_score.black_twice == 98);
    assert(territory_score.white_twice == 15);
}

void test_encoding_and_symmetry() {
    GoState state(rules());
    state.apply(0);
    state.apply(8);
    const auto encoding = az::v2::games::go::canonical_encoding(state);
    assert(encoding.planes == 9);
    assert(encoding.at(0, 0, 0) == 1);
    assert(encoding.at(1, 1, 1) == 1);
    assert(encoding.at(2, 0, 0) == 1);
    assert(encoding.at(8, 0, 0) == 1);

    constexpr std::array symmetries{
        Symmetry::Identity,         Symmetry::Rotate90,         Symmetry::Rotate180,
        Symmetry::Rotate270,        Symmetry::Reflect,          Symmetry::ReflectRotate90,
        Symmetry::ReflectRotate180, Symmetry::ReflectRotate270,
    };
    for (const Symmetry symmetry : symmetries) {
        for (std::int32_t action = 0; action < state.action_count(); ++action) {
            const std::int32_t transformed =
                az::v2::games::go::transform_action(action, state.board_size(), symmetry);
            const std::int32_t restored = az::v2::games::go::transform_action(
                transformed, state.board_size(), az::v2::games::go::inverse_symmetry(symmetry));
            assert(restored == action);
        }
        const auto transformed = az::v2::games::go::transform_encoding(encoding, symmetry);
        const auto restored = az::v2::games::go::transform_encoding(
            transformed, az::v2::games::go::inverse_symmetry(symmetry));
        assert(restored == encoding);
    }
}

void test_invalid_inputs() {
    bool rejected_size = false;
    try {
        GoState invalid(rules(8));
    } catch (const std::invalid_argument &) {
        rejected_size = true;
    }
    assert(rejected_size);

    bool rejected_extreme_size = false;
    try {
        GoState invalid(GoRules{.board_size = std::numeric_limits<std::int32_t>::max(),
                                .komi_half_points = 15,
                                .safety_ply_cap = std::numeric_limits<std::int32_t>::max(),
                                .history_length = 4});
    } catch (const std::invalid_argument &) {
        rejected_extreme_size = true;
    }
    assert(rejected_extreme_size);

    bool rejected_small_cap = false;
    try {
        GoState invalid(rules(7, 48));
    } catch (const std::invalid_argument &) {
        rejected_small_cap = true;
    }
    assert(rejected_small_cap);

    bool rejected_extreme_history = false;
    try {
        GoState invalid(GoRules{.board_size = 7,
                                .komi_half_points = 15,
                                .safety_ply_cap = 200,
                                .history_length = std::numeric_limits<std::int32_t>::max()});
    } catch (const std::invalid_argument &) {
        rejected_extreme_history = true;
    }
    assert(rejected_extreme_history);

    const auto empty = board_with({});
    bool rejected_extreme_ply = false;
    try {
        (void) GoState::restore(rules(), empty, Player::White,
                                std::numeric_limits<std::int32_t>::max(), 0, {empty});
    } catch (const std::invalid_argument &) {
        rejected_extreme_ply = true;
    }
    assert(rejected_extreme_ply);

    GoState beyond_cap_source(rules());
    for (std::int32_t ply = 0; ply < 50; ++ply) {
        const auto legal = beyond_cap_source.legal_actions();
        const auto placement =
            std::find_if(legal.begin(), legal.end(), [&beyond_cap_source](std::int32_t action) {
                return action != beyond_cap_source.pass_action();
            });
        assert(placement != legal.end());
        beyond_cap_source.apply(*placement);
    }
    bool rejected_ply_beyond_cap = false;
    try {
        (void) GoState::restore(rules(7, 49), beyond_cap_source.board(),
                                beyond_cap_source.current_player(), beyond_cap_source.ply(),
                                beyond_cap_source.consecutive_passes(),
                                beyond_cap_source.position_history());
    } catch (const std::invalid_argument &) {
        rejected_ply_beyond_cap = true;
    }
    assert(rejected_ply_beyond_cap);

    bool rejected_transform_size = false;
    try {
        (void) az::v2::games::go::transform_action(0, std::numeric_limits<std::int32_t>::max(),
                                                   Symmetry::Identity);
    } catch (const std::invalid_argument &) {
        rejected_transform_size = true;
    }
    assert(rejected_transform_size);

    bool rejected_symmetry = false;
    try {
        (void) az::v2::games::go::inverse_symmetry(static_cast<Symmetry>(127));
    } catch (const std::invalid_argument &) {
        rejected_symmetry = true;
    }
    assert(rejected_symmetry);

    GoState state(rules());
    assert(!state.is_legal(-1));
    assert(!state.is_legal(state.action_count()));
    state.apply(0);
    assert(!state.is_legal(0));
}

} // namespace

int main() {
    test_configuration_and_passes();
    test_capture_and_suicide();
    test_positional_superko();
    test_copy_hash_and_cap();
    test_restored_pass_invariants();
    test_area_scoring_with_integer_komi();
    test_encoding_and_symmetry();
    test_invalid_inputs();
}
