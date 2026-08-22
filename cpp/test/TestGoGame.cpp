#include "TestRunner.hpp"
#include "games/go/GoGame.hpp"
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
GoBoard<BoardSize> boardWith(const std::vector<int> &black, const std::vector<int> &white) {
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
std::array<GoBoard<BoardSize>, 8> historyWithCurrent(const GoBoard<BoardSize> &board) {
    std::array<GoBoard<BoardSize>, 8> history{};
    history[0] = board;
    return history;
}

template <std::size_t BoardSize> void testInitialAndActions() {
    using Contract = GoGame<BoardSize, 8>;
    const GoRules rules{
        .komi_half_points = 15,
        .maximum_moves = static_cast<int>(BoardSize * BoardSize * 4),
    };
    const typename Contract::State initial(rules);
    require(initial.player() == GoPlayer::black, "Black must move first");
    require(initial.board().black.none() && initial.board().white.none(),
            "Initial Go board must be empty");
    for (const auto &historicBoard : initial.history()) {
        require(historicBoard.black.none() && historicBoard.white.none(),
                "Unavailable Go history must be zero initialized");
    }
    const auto legal = Contract::legalActions(initial);
    require(legal.size() == BoardSize * BoardSize + 1, "Every initial Go action must be legal");
    require(legal.back().isPass(), "Pass must be the final Go action");
    require(legal.front().point() == BitBoard<BoardSize>::point(0),
            "Go action point must decode its board coordinate");
    for (const auto action : legal) {
        require(GoAction<BoardSize>(Contract::Encoding::actionId(action, initial)) == action,
                "Go action encoding must round trip");
    }
}

void testCaptureSuicideAndKo() {
    using Contract = Go7Game;
    using Position = Contract::State;
    const GoRules rules{.komi_half_points = 15, .maximum_moves = 200};

    const auto captureBoard = boardWith<7>({1, 7, 9}, {8});
    const Position capture = Position::restore(historyWithCurrent(captureBoard), GoPlayer::black,
                                               std::nullopt, 0, 6, rules);
    const Position captured = capture.child(GoAction<7>(15));
    require(!captured.board().white.test(8), "Surrounded Go stone must be captured");
    require(captured.board().black.test(15), "Capturing stone must be placed");

    const auto suicideBoard = boardWith<7>({1, 7, 9, 15}, {});
    const Position suicide = Position::restore(historyWithCurrent(suicideBoard), GoPlayer::white,
                                               std::nullopt, 0, 4, rules);
    require(!suicide.isLegal(GoAction<7>(8)), "Suicide must be illegal");

    const auto koBoard = boardWith<7>({0, 2}, {1, 7, 9, 15});
    const Position beforeKo =
        Position::restore(historyWithCurrent(koBoard), GoPlayer::black, std::nullopt, 0, 8, rules);
    const Position afterKo = beforeKo.child(GoAction<7>(8));
    require(afterKo.koPoint() == BitBoard<7>::point(1), "Single-stone ko point must be recorded");
    require(!afterKo.isLegal(GoAction<7>(1)), "Immediate simple-ko recapture must be illegal");
    const Position afterPass = afterKo.child(GoAction<7>::pass());
    require(!afterPass.koPoint().has_value(), "Pass must clear the simple-ko point");
    require(afterPass.isLegal(GoAction<7>(1)),
            "Ko recapture must become legal after an intervening move");
}

void testHistoryEncodingAndHash() {
    using Contract = Go7Game;
    const GoRules rules{.komi_half_points = 15, .maximum_moves = 200};
    const typename Contract::State initial(rules);
    const auto blackPlayed = initial.child(GoAction<7>(0));
    const auto whitePlayed = blackPlayed.child(GoAction<7>(8));
    const auto whiteToMoveEncoding = encodeGoPosition(blackPlayed);
    require(whiteToMoveEncoding.binaryPlanes[0].none() &&
                whiteToMoveEncoding.binaryPlanes[1].test(0) &&
                whiteToMoveEncoding.scalarPlanes[0] == 0,
            "Go encoding must convert absolute history to the current-player perspective");
    require(whitePlayed.history()[0].black.test(0) && whitePlayed.history()[0].white.test(8),
            "Current Go history board is incorrect");
    require(whitePlayed.history()[1].black.test(0) && whitePlayed.history()[1].white.none(),
            "Go history must shift older boards back one slot");
    require(whitePlayed.history()[2].black.none() && whitePlayed.history()[2].white.none(),
            "Initial historic board must remain empty");
    require(whitePlayed.hash() == whitePlayed.hash() && whitePlayed.hash() != blackPlayed.hash(),
            "Go hashes must be stable and state-sensitive");

    const auto encoded = encodeGoPosition(whitePlayed);
    require(encoded.binaryPlanes[0].test(0), "Current-player black stone plane is incorrect");
    require(encoded.binaryPlanes[1].test(8), "Opponent white stone plane is incorrect");
    require(encoded.binaryPlanes[2].test(0), "Older current-player plane is incorrect");
    require(encoded.scalarPlanes[0] == 1, "Black-to-move scalar plane is incorrect");
    std::array<std::int8_t, decltype(encoded)::packedBytes> packed{};
    encoded.writePackedInto(packed);
    require((static_cast<std::uint8_t>(packed[0]) & 1U) != 0,
            "Packed Go encoding must use canonical point mapping");
    std::array<std::int8_t, GoRepresentationDimensions<7>::channelCount * 49> tensor{};
    Contract::Encoding::encodeInputInto(whitePlayed, tensor.data());
    require(tensor[0] == 1 && tensor[49 + 8] == 1,
            "Expanded Go tensor must match packed plane semantics");
}

void testTerminationAndScoring() {
    using Position = Go7Game::State;
    const GoRules rules{.komi_half_points = 15, .maximum_moves = 200};
    const auto scoringBoard = boardWith<7>({1, 7}, {});
    const Position scoring = Position::restore(historyWithCurrent(scoringBoard), GoPlayer::black,
                                               std::nullopt, 0, 2, rules);
    require(scoring.areaScore() == GoAreaScore{.black_half_points = 98, .white_half_points = 15},
            "Go area scoring must include surrounded territory and komi");

    const auto once = scoring.child(GoAction<7>::pass());
    const auto twice = once.child(GoAction<7>::pass());
    require(twice.terminationReason() == GoTerminationReason::two_passes,
            "Two passes must terminate Go");
    require(twice.terminalResult().score == scoring.areaScore(),
            "Two-pass result must contain the final area score");
    require(Go7Game::terminalValue(twice) == 1.0F,
            "Terminal value must use the side-to-move perspective");
    require(twice.legalActions().empty(), "Terminal Go positions must have no legal actions");

    const GoRules cappedRules{.komi_half_points = 15, .maximum_moves = 49};
    const Position capped = Position::restore(historyWithCurrent(GoBoard<7>{}), GoPlayer::white,
                                              std::nullopt, 0, 49, cappedRules);
    require(capped.terminationReason() == GoTerminationReason::maximum_moves,
            "Maximum move bound must terminate Go");
    require(capped.terminalResult().score ==
                GoAreaScore{.black_half_points = 0, .white_half_points = 15},
            "Maximum-move adjudication must use the configured area score");
    require(Go7Game::terminalValue(capped) == 1.0F,
            "Maximum-move adjudication must produce a value from the side-to-move perspective");
}

void testInvalidBoundaries() {
    bool rejectedRules = false;
    try {
        (void) GoPosition<7>(GoRules{.komi_half_points = 15, .maximum_moves = 48});
    } catch (const std::invalid_argument &) {
        rejectedRules = true;
    }
    require(rejectedRules, "Maximum move bound below board area must be rejected");

    bool rejectedAction = false;
    try {
        (void) GoAction<7>(50);
    } catch (const std::invalid_argument &) {
        rejectedAction = true;
    }
    require(rejectedAction, "Out-of-range Go action must be rejected");
}

} // namespace

int runGoGameTests() {
    try {
        testInitialAndActions<7>();
        testInitialAndActions<9>();
        testInitialAndActions<13>();
        testInitialAndActions<19>();
        testCaptureSuicideAndKo();
        testHistoryEncodingAndHash();
        testTerminationAndScoring();
        testInvalidBoundaries();
        std::cout << "Go game tests passed\n";
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
