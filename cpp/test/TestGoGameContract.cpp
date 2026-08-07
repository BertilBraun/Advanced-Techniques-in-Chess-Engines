#include "TestRunner.hpp"
#include "games/go/GoGameContract.hpp"
#include "util/py.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <std::size_t BoardSize>
GoBoard<BoardSize> board_with(const std::vector<int> &black, const std::vector<int> &white) {
    GoBoard<BoardSize> board;
    for (const int point : black) {
        board.black.set(static_cast<std::size_t>(point));
    }
    for (const int point : white) {
        board.white.set(static_cast<std::size_t>(point));
    }
    return board;
}

template <std::size_t BoardSize>
std::array<GoBoard<BoardSize>, 8> history_with_current(const GoBoard<BoardSize> &board) {
    std::array<GoBoard<BoardSize>, 8> history{};
    history[0] = board;
    return history;
}

template <std::size_t BoardSize> void test_initial_and_actions() {
    using Contract = GoGameContract<BoardSize, 8>;
    const GoRules rules{
        .komi_half_points = 15,
        .maximum_moves = static_cast<int>(BoardSize * BoardSize * 4),
    };
    const auto initial = Contract::initialPosition(rules);
    require(initial.player() == GoPlayer::black, "Black must move first");
    require(initial.board().black.none() && initial.board().white.none(),
            "Initial Go board must be empty");
    for (const auto &historic_board : initial.history()) {
        require(historic_board.black.none() && historic_board.white.none(),
                "Unavailable Go history must be zero initialized");
    }
    const auto legal = Contract::legalActions(initial);
    require(legal.size() == BoardSize * BoardSize + 1, "Every initial Go action must be legal");
    require(legal.back().is_pass(), "Pass must be the final Go action");
    require(legal.front().point() == BitBoard<BoardSize>::point(0),
            "Go action point must decode its board coordinate");
    for (const auto action : legal) {
        const int action_id = Contract::actionId(action, initial);
        const auto decoded = Contract::decodeActions({action_id}, initial);
        require(decoded.size() == 1 && decoded.front() == action,
                "Go action encoding must round trip");
    }
}

void test_capture_suicide_and_ko() {
    using Contract = Go7GameContract;
    using Position = Contract::Position;
    const GoRules rules{.komi_half_points = 15, .maximum_moves = 200};

    const auto capture_board = board_with<7>({1, 7, 9}, {8});
    const Position capture = Position::restore(history_with_current(capture_board), GoPlayer::black,
                                               std::nullopt, 0, 6, rules);
    const Position captured = capture.child(GoAction<7>(15));
    require(!captured.board().white.test(8), "Surrounded Go stone must be captured");
    require(captured.board().black.test(15), "Capturing stone must be placed");

    const auto suicide_board = board_with<7>({1, 7, 9, 15}, {});
    const Position suicide = Position::restore(history_with_current(suicide_board), GoPlayer::white,
                                               std::nullopt, 0, 4, rules);
    require(!suicide.is_legal(GoAction<7>(8)), "Suicide must be illegal");

    const auto ko_board = board_with<7>({0, 2}, {1, 7, 9, 15});
    const Position before_ko = Position::restore(history_with_current(ko_board), GoPlayer::black,
                                                 std::nullopt, 0, 8, rules);
    const Position after_ko = before_ko.child(GoAction<7>(8));
    require(after_ko.ko_point() == BitBoard<7>::point(1), "Single-stone ko point must be recorded");
    require(!after_ko.is_legal(GoAction<7>(1)), "Immediate simple-ko recapture must be illegal");
    const Position after_pass = after_ko.child(GoAction<7>::pass());
    require(!after_pass.ko_point().has_value(), "Pass must clear the simple-ko point");
    require(after_pass.is_legal(GoAction<7>(1)),
            "Ko recapture must become legal after an intervening move");
}

void test_history_encoding_hash_and_symmetries() {
    using Contract = Go7GameContract;
    const GoRules rules{.komi_half_points = 15, .maximum_moves = 200};
    const auto initial = Contract::initialPosition(rules);
    const auto black_played = initial.child(GoAction<7>(0));
    const auto white_played = black_played.child(GoAction<7>(8));
    const auto white_to_move_encoding = Contract::encodeInput(black_played);
    require(white_to_move_encoding.binary_planes[0].none() &&
                white_to_move_encoding.binary_planes[1].test(0) &&
                white_to_move_encoding.scalar_planes[0] == 0,
            "Go encoding must convert absolute history to the current-player perspective");
    require(white_played.history()[0].black.test(0) && white_played.history()[0].white.test(8),
            "Current Go history board is incorrect");
    require(white_played.history()[1].black.test(0) && white_played.history()[1].white.none(),
            "Go history must shift older boards back one slot");
    require(white_played.history()[2].black.none() && white_played.history()[2].white.none(),
            "Initial historic board must remain empty");
    require(white_played.hash() == white_played.hash() &&
                white_played.hash() != black_played.hash(),
            "Go hashes must be stable and state-sensitive");

    const auto encoded = Contract::encodeInput(white_played);
    require(encoded.binary_planes[0].test(0), "Current-player black stone plane is incorrect");
    require(encoded.binary_planes[1].test(8), "Opponent white stone plane is incorrect");
    require(encoded.binary_planes[2].test(0), "Older current-player plane is incorrect");
    require(encoded.scalar_planes[0] == 1, "Black-to-move scalar plane is incorrect");
    std::array<std::int8_t, decltype(encoded)::packed_bytes> packed{};
    Contract::writePackedInput(encoded, packed.data());
    require((static_cast<std::uint8_t>(packed[0]) & 1U) != 0,
            "Packed Go encoding must use canonical point mapping");
    std::array<std::int8_t, GoRepresentationDimensions<7>::channel_count * 49> tensor{};
    Contract::encodeInputInto(white_played, tensor.data());
    require(tensor[0] == 1 && tensor[49 + 8] == 1,
            "Expanded Go tensor must match packed plane semantics");

    constexpr std::array symmetries{
        GoSymmetry::identity,
        GoSymmetry::rotate_90,
        GoSymmetry::rotate_180,
        GoSymmetry::rotate_270,
        GoSymmetry::reflect,
        GoSymmetry::reflect_rotate_90,
        GoSymmetry::reflect_rotate_180,
        GoSymmetry::reflect_rotate_270,
    };
    for (const GoSymmetry symmetry : symmetries) {
        for (const int action_id : range(GoAction<7>::action_count)) {
            const GoAction<7> action(action_id);
            const auto transformed = Contract::transformAction(action, symmetry);
            const auto restored =
                Contract::transformAction(transformed, Contract::inverseSymmetry(symmetry));
            require(restored == action, "Go action symmetry must be invertible");
        }
        const auto transformed = Contract::transformEncoding(encoded, symmetry);
        const auto restored =
            Contract::transformEncoding(transformed, Contract::inverseSymmetry(symmetry));
        require(restored == encoded, "Go encoding symmetry must be invertible");
    }
}

void test_termination_and_scoring() {
    using Position = Go7GameContract::Position;
    const GoRules rules{.komi_half_points = 15, .maximum_moves = 200};
    const auto scoring_board = board_with<7>({1, 7}, {});
    const Position scoring = Position::restore(history_with_current(scoring_board), GoPlayer::black,
                                               std::nullopt, 0, 2, rules);
    require(scoring.area_score() == GoAreaScore{.black_half_points = 98, .white_half_points = 15},
            "Go area scoring must include surrounded territory and komi");

    const auto once = scoring.child(GoAction<7>::pass());
    const auto twice = once.child(GoAction<7>::pass());
    require(twice.termination_reason() == GoTerminationReason::two_passes,
            "Two passes must terminate Go");
    require(twice.terminal_result().score == scoring.area_score(),
            "Two-pass result must contain the final area score");
    require(Go7GameContract::terminalValue(twice) == 1.0F,
            "Terminal value must use the side-to-move perspective");
    require(twice.legal_actions().empty(), "Terminal Go positions must have no legal actions");

    const GoRules capped_rules{.komi_half_points = 15, .maximum_moves = 49};
    const Position capped = Position::restore(history_with_current(GoBoard<7>{}), GoPlayer::white,
                                              std::nullopt, 0, 49, capped_rules);
    require(capped.termination_reason() == GoTerminationReason::maximum_moves,
            "Maximum move bound must terminate Go");
    require(!capped.terminal_result().score.has_value() &&
                !Go7GameContract::terminalValue(capped).has_value(),
            "Maximum-move termination must not invent a score or value");
}

void test_invalid_boundaries() {
    bool rejected_rules = false;
    try {
        (void) Go7GameContract::initialPosition(
            GoRules{.komi_half_points = 15, .maximum_moves = 48});
    } catch (const std::invalid_argument &) {
        rejected_rules = true;
    }
    require(rejected_rules, "Maximum move bound below board area must be rejected");

    bool rejected_action = false;
    try {
        (void) GoAction<7>(50);
    } catch (const std::invalid_argument &) {
        rejected_action = true;
    }
    require(rejected_action, "Out-of-range Go action must be rejected");
}

} // namespace

int runGoGameContractTests() {
    try {
        test_initial_and_actions<7>();
        test_initial_and_actions<9>();
        test_initial_and_actions<13>();
        test_initial_and_actions<19>();
        test_capture_suicide_and_ko();
        test_history_encoding_hash_and_symmetries();
        test_termination_and_scoring();
        test_invalid_boundaries();
        std::cout << "Go game contract tests passed\n";
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
