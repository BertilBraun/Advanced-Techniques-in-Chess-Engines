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

using az::games::go::GoRules;
using az::games::go::GoState;
using az::games::go::GoSymmetryOperations;
using az::games::go::Player;
using az::games::go::Stone;
using az::games::go::Symmetry;
using az::games::go::TerminationReason;

static_assert(az::games::GameState<GoState>);
static_assert(az::games::GameSymmetry<GoSymmetryOperations>);

namespace {

GoRules rules(int32 size = 7, int32 cap = 200) {
    return GoRules{
        .boardSize = size,
        .komiHalfPoints = 15,
        .safetyPlyCap = cap,
        .historyLength = 4,
    };
}

std::vector<Stone> boardWith(std::initializer_list<std::pair<int32, Stone>> stones) {
    std::vector<Stone> board(49, Stone::Empty);
    for (const auto &[point, stone] : stones) {
        board[static_cast<std::size_t>(point)] = stone;
    }
    return board;
}

void testConfigurationAndPasses() {
    for (const int32 size : {3, 5, 7, 9, 13}) {
        GoState state(rules(size));
        assert(state.actionCount() == size * size + 1);
        assert(state.passAction() == size * size);
        assert(state.legalActions().size() == static_cast<std::size_t>(size * size + 1));
        state.apply(state.passAction());
        assert(!state.isTerminal());
        assert(state.isLegal(state.passAction()));
        state.apply(state.passAction());
        assert(state.isTerminal());
        assert(state.terminationReason() == TerminationReason::TwoPasses);
        assert(state.legalActions().empty());
        assert(state.terminalResult().score.has_value());
        assert(state.terminalResult().winner == Player::White);
    }
}

void testCaptureAndSuicide() {
    const auto captureBoard =
        boardWith({{8, Stone::White}, {1, Stone::Black}, {7, Stone::Black}, {9, Stone::Black}});
    GoState capture = GoState::restore(rules(), captureBoard, Player::Black, 0, 0, {captureBoard});
    capture.apply(15);
    assert(capture.board()[8] == Stone::Empty);

    const auto multiCaptureBoard = boardWith({{8, Stone::White},
                                              {9, Stone::White},
                                              {1, Stone::Black},
                                              {2, Stone::Black},
                                              {7, Stone::Black},
                                              {10, Stone::Black},
                                              {15, Stone::Black}});
    GoState multiCapture =
        GoState::restore(rules(), multiCaptureBoard, Player::Black, 0, 0, {multiCaptureBoard});
    multiCapture.apply(16);
    assert(multiCapture.board()[8] == Stone::Empty);
    assert(multiCapture.board()[9] == Stone::Empty);

    const auto suicideBoard =
        boardWith({{1, Stone::Black}, {2, Stone::White}, {7, Stone::White}, {8, Stone::White}});
    const GoState suicide =
        GoState::restore(rules(), suicideBoard, Player::Black, 0, 0, {suicideBoard});
    assert(!suicide.isLegal(0));
}

void testPositionalSuperko() {
    const auto before = boardWith({{8, Stone::White},
                                   {14, Stone::White},
                                   {16, Stone::White},
                                   {22, Stone::White},
                                   {1, Stone::Black},
                                   {7, Stone::Black},
                                   {9, Stone::Black}});
    GoState ko = GoState::restore(rules(), before, Player::Black, 0, 0, {before});
    ko.apply(15);
    assert(!ko.isLegal(8));

    const auto repeated = boardWith({{14, Stone::White},
                                     {16, Stone::White},
                                     {22, Stone::White},
                                     {1, Stone::Black},
                                     {7, Stone::Black},
                                     {9, Stone::Black},
                                     {15, Stone::Black}});
    const auto middle = boardWith({{30, Stone::White}});
    const GoState longerCycle =
        GoState::restore(rules(), before, Player::Black, 2, 0, {repeated, middle, before});
    assert(!longerCycle.isLegal(15));
}

void testCopyHashAndCap() {
    GoState state(rules());
    assert(state.stateHash() == 6493982775080899741ULL);
    state.apply(0);
    GoState copy = state;
    assert(copy == state);
    assert(copy.stateHash() == state.stateHash());
    copy.apply(1);
    assert(!(copy == state));
    assert(copy.stateHash() != state.stateHash());
    GoState longGame(rules());
    for (int32 ply = 0; ply < 48; ++ply) {
        const auto legal = longGame.legalActions();
        const auto placement = std::find_if(legal.begin(), legal.end(), [&longGame](int32 action) {
            return action != longGame.passAction();
        });
        assert(placement != legal.end());
        longGame.apply(*placement);
    }
    GoState capped =
        GoState::restore(rules(7, 49), longGame.board(), longGame.currentPlayer(), longGame.ply(),
                         longGame.consecutivePasses(), longGame.positionHistory());
    capped.apply(capped.passAction());
    assert(capped.terminationReason() == TerminationReason::SafetyPlyCap);
    assert(!capped.terminalResult().score.has_value());
    assert(!capped.terminalResult().winner.has_value());
}

void testRestoredPassInvariants() {
    const auto empty = boardWith({});
    const auto placed = boardWith({{0, Stone::Black}});
    const GoState noPass = GoState::restore(rules(), placed, Player::White, 1, 0, {empty, placed});
    const GoState onePass =
        GoState::restore(rules(), placed, Player::Black, 2, 1, {empty, placed, placed});
    const GoState twoPasses =
        GoState::restore(rules(), placed, Player::White, 3, 2, {empty, placed, placed, placed});
    assert(noPass.consecutivePasses() == 0);
    assert(onePass.consecutivePasses() == 1);
    assert(twoPasses.terminationReason() == TerminationReason::TwoPasses);

    bool rejectedZeroWithDuplicate = false;
    try {
        (void) GoState::restore(rules(), empty, Player::White, 1, 0, {empty, empty});
    } catch (const std::invalid_argument &) {
        rejectedZeroWithDuplicate = true;
    }
    assert(rejectedZeroWithDuplicate);

    bool rejectedSingleAfterPass = false;
    try {
        (void) GoState::restore(rules(), placed, Player::White, 3, 1,
                                {empty, placed, placed, placed});
    } catch (const std::invalid_argument &) {
        rejectedSingleAfterPass = true;
    }
    assert(rejectedSingleAfterPass);

    bool rejectedIncompleteDoublePass = false;
    try {
        (void) GoState::restore(rules(), placed, Player::Black, 2, 2, {empty, placed, placed});
    } catch (const std::invalid_argument &) {
        rejectedIncompleteDoublePass = true;
    }
    assert(rejectedIncompleteDoublePass);
}

void testAreaScoringWithIntegerKomi() {
    const auto neutralBoard = boardWith({{0, Stone::Black}, {48, Stone::White}});
    const GoState state =
        GoState::restore(rules(), neutralBoard, Player::Black, 0, 0, {neutralBoard});
    const auto score = state.areaScore();
    assert(score.blackTwice == 2);
    assert(score.whiteTwice == 17);
    assert(score.winner() == Player::White);

    const auto surroundedCorner = boardWith({{1, Stone::Black}, {7, Stone::Black}});
    const GoState territory =
        GoState::restore(rules(), surroundedCorner, Player::Black, 0, 0, {surroundedCorner});
    const auto territoryScore = territory.areaScore();
    assert(territoryScore.blackTwice == 98);
    assert(territoryScore.whiteTwice == 15);
}

void testEncodingAndSymmetry() {
    GoState state(rules());
    state.apply(0);
    state.apply(8);
    const auto encoding = az::games::go::canonicalEncoding(state);
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
        for (int32 action = 0; action < state.actionCount(); ++action) {
            const int32 transformed =
                az::games::go::transformAction(action, state.boardSize(), symmetry);
            const int32 restored = az::games::go::transformAction(
                transformed, state.boardSize(), az::games::go::inverseSymmetry(symmetry));
            assert(restored == action);
        }
        const auto transformed = az::games::go::transformEncoding(encoding, symmetry);
        const auto restored =
            az::games::go::transformEncoding(transformed, az::games::go::inverseSymmetry(symmetry));
        assert(restored == encoding);
    }
}

void testInvalidInputs() {
    constexpr int32 largestRepresentableBoardSize = 46'340;
    const int32 largestPassAction = largestRepresentableBoardSize * largestRepresentableBoardSize;
    const az::games::go::GoEncodingShape largestShape =
        az::games::go::checkedEncodingShape(2, largestRepresentableBoardSize);
    assert(largestShape.planeSize == static_cast<std::size_t>(largestPassAction));
    assert(largestShape.totalSize == static_cast<std::size_t>(largestPassAction) * 2U);
    assert(largestShape.index(1, largestRepresentableBoardSize - 1,
                              largestRepresentableBoardSize - 1) == largestShape.totalSize - 1U);
    if constexpr (sizeof(std::size_t) == sizeof(uint32)) {
        bool rejectedUnrepresentableEncodingSize = false;
        try {
            (void) az::games::go::checkedEncodingShape(3, largestRepresentableBoardSize);
        } catch (const std::length_error &) {
            rejectedUnrepresentableEncodingSize = true;
        }
        assert(rejectedUnrepresentableEncodingSize);
    } else {
        const az::games::go::GoEncodingShape maximumPlaneShape =
            az::games::go::checkedEncodingShape(std::numeric_limits<int32>::max(),
                                                largestRepresentableBoardSize);
        assert(maximumPlaneShape.totalSize ==
               static_cast<std::size_t>(std::numeric_limits<int32>::max()) *
                   static_cast<std::size_t>(largestPassAction));
    }
    assert(az::games::go::transformAction(largestPassAction, largestRepresentableBoardSize,
                                          Symmetry::Rotate90) == largestPassAction);

    const az::games::go::GoEncoding malformedEncoding{
        .planes = 2,
        .boardSize = largestRepresentableBoardSize,
        .values = {},
    };
    bool rejectedMalformedEncoding = false;
    try {
        (void) malformedEncoding.at(1, largestRepresentableBoardSize - 1,
                                    largestRepresentableBoardSize - 1);
    } catch (const std::invalid_argument &) {
        rejectedMalformedEncoding = true;
    }
    assert(rejectedMalformedEncoding);

    bool rejectedSize = false;
    try {
        GoState invalid(rules(2));
    } catch (const std::invalid_argument &) {
        rejectedSize = true;
    }
    assert(rejectedSize);

    bool rejectedExtremeSize = false;
    try {
        GoState invalid(GoRules{.boardSize = std::numeric_limits<int32>::max(),
                                .komiHalfPoints = 15,
                                .safetyPlyCap = std::numeric_limits<int32>::max(),
                                .historyLength = 4});
    } catch (const std::invalid_argument &) {
        rejectedExtremeSize = true;
    }
    assert(rejectedExtremeSize);

    bool rejectedSmallCap = false;
    try {
        GoState invalid(rules(7, 48));
    } catch (const std::invalid_argument &) {
        rejectedSmallCap = true;
    }
    assert(rejectedSmallCap);

    bool rejectedExtremeHistory = false;
    try {
        GoState invalid(GoRules{.boardSize = 7,
                                .komiHalfPoints = 15,
                                .safetyPlyCap = 200,
                                .historyLength = std::numeric_limits<int32>::max()});
    } catch (const std::invalid_argument &) {
        rejectedExtremeHistory = true;
    }
    assert(rejectedExtremeHistory);

    const auto empty = boardWith({});
    bool rejectedExtremePly = false;
    try {
        (void) GoState::restore(rules(), empty, Player::White, std::numeric_limits<int32>::max(), 0,
                                {empty});
    } catch (const std::invalid_argument &) {
        rejectedExtremePly = true;
    }
    assert(rejectedExtremePly);

    GoState beyondCapSource(rules());
    for (int32 ply = 0; ply < 50; ++ply) {
        const auto legal = beyondCapSource.legalActions();
        const auto placement =
            std::find_if(legal.begin(), legal.end(), [&beyondCapSource](int32 action) {
                return action != beyondCapSource.passAction();
            });
        assert(placement != legal.end());
        beyondCapSource.apply(*placement);
    }
    bool rejectedPlyBeyondCap = false;
    try {
        (void) GoState::restore(rules(7, 49), beyondCapSource.board(),
                                beyondCapSource.currentPlayer(), beyondCapSource.ply(),
                                beyondCapSource.consecutivePasses(),
                                beyondCapSource.positionHistory());
    } catch (const std::invalid_argument &) {
        rejectedPlyBeyondCap = true;
    }
    assert(rejectedPlyBeyondCap);

    bool rejectedTransformSize = false;
    try {
        (void) az::games::go::transformAction(0, largestRepresentableBoardSize + 1,
                                              Symmetry::Identity);
    } catch (const std::invalid_argument &) {
        rejectedTransformSize = true;
    }
    assert(rejectedTransformSize);

    bool rejectedSymmetry = false;
    try {
        (void) az::games::go::inverseSymmetry(static_cast<Symmetry>(127));
    } catch (const std::invalid_argument &) {
        rejectedSymmetry = true;
    }
    assert(rejectedSymmetry);

    GoState state(rules());
    assert(!state.isLegal(-1));
    assert(!state.isLegal(state.actionCount()));
    state.apply(0);
    assert(!state.isLegal(0));
}

} // namespace

int main() {
    testConfigurationAndPasses();
    testCaptureAndSuicide();
    testPositionalSuperko();
    testCopyHashAndCap();
    testRestoredPassInvariants();
    testAreaScoringWithIntegerKomi();
    testEncodingAndSymmetry();
    testInvalidInputs();
}
