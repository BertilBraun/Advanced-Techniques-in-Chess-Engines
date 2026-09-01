#include "TestRunner.hpp"
#include "games/chess/ChessGame.hpp"
#include "games/go/GoGame.hpp"
#include "search/InferencePipeline.hpp"
#include "util/py.hpp"

#include "bitboard.h"
#include "position.h"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr float scoreTolerance = 1e-6F;
constexpr std::array<float, 3> validOutcome = {0.25F, 0.5F, 0.25F};

void require(const bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Operation>
void requireInvalidArgument(Operation operation, const std::string &message) {
    bool rejected = false;
    try {
        operation();
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, message);
}

void requireLegalRoundTrips(const Board &board, const std::string &description) {
    std::vector<int> actionIds;
    actionIds.reserve(board.validMoves().size());
    for (const Stockfish::Move move : board.validMoves()) {
        const ChessAction action(move);
        const int actionId = ChessEncoding::actionId(action, board);
        require(ChessEncoding::decodeAction(actionId, board) == action,
                description + ": legal action did not round trip");
        actionIds.push_back(actionId);
    }
    std::ranges::sort(actionIds);
    require(std::ranges::adjacent_find(actionIds) == actionIds.end(),
            description + ": legal actions did not have unique ids");
}

std::vector<float> finiteLogits() {
    std::vector<float> policy(ChessEncoding::actionCount);
    for (const int actionId : range(ChessEncoding::actionCount)) {
        policy[actionId] =
            1000.0F + static_cast<float>(actionId) / static_cast<float>(ChessEncoding::actionCount);
    }
    return policy;
}

void requireNormalized(const Board &board, const std::vector<float> &policy,
                       const std::string &description) {
    const SearchInferenceResult<ChessGame> result =
        processInferencePosition<ChessGame>(policy.data(), validOutcome.data(), board);
    float sum = 0.0F;
    int previousActionId = -1;
    for (const auto &[action, actionId, probability] : result.actions) {
        static_cast<void>(action);
        require(actionId > previousActionId, description + ": actions are not ordered by id");
        require(probability >= 0.0F, description + ": returned a negative probability");
        previousActionId = actionId;
        sum += probability;
    }
    require(std::abs(sum - 1.0F) <= scoreTolerance, description + ": policy is not normalized");
}

void testRepresentativePositions() {
    const Board initial;
    requireNormalized(initial, finiteLogits(), "starting position");
    requireNormalized(Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"),
                      finiteLogits(), "black-to-move position");
    const Board promotion("7k/P7/8/8/8/8/8/7K w - - 0 1");
    require(std::ranges::count_if(promotion.validMoves(),
                                  [](const Stockfish::Move move) {
                                      return move.type_of() == Stockfish::PROMOTION;
                                  }) == 4,
            "promotion position did not contain four promotion choices");
    requireNormalized(promotion, finiteLogits(), "promotion position");
}

void testSelectedActionSpaceLayout() {
    if constexpr (chessActionEncoding == ChessActionEncoding::Reduced) {
        require(ChessEncoding::actionCount == ChessRepresentationDimensions::reducedActionCount,
                "chess action space is not the reduced move layout");
    } else {
        require(ChessEncoding::actionCount == ChessRepresentationDimensions::policyPlaneActionCount,
                "chess action space is not the direct policy-plane layout");
    }
    for (const int actionId : range(ChessEncoding::actionCount)) {
        require(ChessEncoding::mirrorActionId(ChessEncoding::mirrorActionId(actionId)) == actionId,
                "chess action mirroring is not an involution");
    }

    const Board promotion("7k/P7/8/8/8/8/8/7K w - - 0 1");
    std::vector<int> promotionIds;
    for (const Stockfish::Move move : promotion.validMoves()) {
        if (move.type_of() == Stockfish::PROMOTION) {
            const int actionId = ChessEncoding::actionId(ChessAction(move), promotion);
            if constexpr (chessActionEncoding == ChessActionEncoding::PolicyPlane) {
                require(actionId % 64 == static_cast<int>(move.from_sq()),
                        "white promotion action does not retain its canonical origin square");
                require(actionId / 64 >= 64,
                        "promotion choices did not use explicit promotion planes");
            }
            promotionIds.push_back(actionId);
        }
    }
    std::ranges::sort(promotionIds);
    require(std::ranges::unique(promotionIds).begin() == promotionIds.end() &&
                promotionIds.size() == 4,
            "promotion choices do not occupy distinct action ids");

    const Board blackPromotion("7K/8/8/8/8/8/p7/7k b - - 0 1");
    for (const Stockfish::Move move : blackPromotion.validMoves()) {
        if (move.type_of() == Stockfish::PROMOTION) {
            const int actionId = ChessEncoding::actionId(ChessAction(move), blackPromotion);
            if constexpr (chessActionEncoding == ChessActionEncoding::PolicyPlane) {
                require(actionId % 64 == static_cast<int>(Stockfish::flip_rank(move.from_sq())),
                        "black promotion action does not retain its canonical origin square");
            }
            require(ChessEncoding::decodeAction(actionId, blackPromotion) == ChessAction(move),
                    "black promotion action did not round-trip through the selected layout");
        }
    }
}

void testReducedActionSpaceIsTotal() {
    if constexpr (chessActionEncoding != ChessActionEncoding::Reduced) {
        return;
    }
    const Board initial;
    for (const int actionId : range(ChessEncoding::actionCount)) {
        try {
            static_cast<void>(ChessEncoding::decodeAction(actionId, initial));
        } catch (const std::invalid_argument &error) {
            require(std::string(error.what()) == "Chess action ID is illegal in this position",
                    "reduced chess action id did not encode an on-board move");
        }
    }
}

void testPlayoutRoundTrips() {
    Board board;
    for (int ply = 0; ply < 400 && !board.isGameOver(); ++ply) {
        requireLegalRoundTrips(board, "playout ply " + std::to_string(ply));
        const std::vector<Stockfish::Move> &moves = board.validMoves();
        const Stockfish::Move move = moves[static_cast<std::size_t>(ply * 7 + 3) % moves.size()];
        const int actionId = ChessEncoding::actionId(ChessAction(move), board);
        const int mirroredActionId = ChessEncoding::mirrorActionId(actionId);
        require(ChessEncoding::mirrorActionId(mirroredActionId) == actionId,
                "playout mirroring is not an involution");
        board.makeMove(move);
    }
}

void testSpecialMoveRoundTrips() {
    const Board initial;
    const Board blackInitial("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
    const Board castling("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    const Board enPassant("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
    const Board whitePromotion("1r5k/P7/8/8/8/8/8/7K w - - 0 1");
    const Board blackPromotion("7k/8/8/8/8/8/p7/1R5K b - - 0 1");

    requireLegalRoundTrips(initial, "initial position");
    requireLegalRoundTrips(blackInitial, "black-to-move initial position");
    requireLegalRoundTrips(castling, "castling position");
    requireLegalRoundTrips(enPassant, "en-passant position");
    requireLegalRoundTrips(whitePromotion, "white promotion-capture position");
    requireLegalRoundTrips(blackPromotion, "black promotion-capture position");

    require(std::ranges::count_if(castling.validMoves(),
                                  [](const Stockfish::Move move) {
                                      return move.type_of() == Stockfish::CASTLING;
                                  }) == 2,
            "castling fixture did not contain both castling actions");
    require(std::ranges::count_if(enPassant.validMoves(),
                                  [](const Stockfish::Move move) {
                                      return move.type_of() == Stockfish::EN_PASSANT;
                                  }) == 1,
            "en-passant fixture did not contain its capture");
    for (const Board *promotion : {&whitePromotion, &blackPromotion}) {
        int straightPromotions = 0;
        int capturePromotions = 0;
        for (const Stockfish::Move move : promotion->validMoves()) {
            if (move.type_of() != Stockfish::PROMOTION) {
                continue;
            }
            if (Stockfish::file_of(move.from_sq()) == Stockfish::file_of(move.to_sq())) {
                ++straightPromotions;
            } else {
                ++capturePromotions;
            }
        }
        require(straightPromotions == 4 && capturePromotions == 4,
                "promotion fixture did not contain all straight and capture choices");
    }
}

void testSemanticMirroring() {
    const Board original("1r5k/P7/8/8/8/8/8/7K w - - 0 1");
    const Board mirrored("k5r1/7P/8/8/8/8/8/K7 w - - 0 1");
    std::vector<int> mirroredActionIds;
    mirroredActionIds.reserve(mirrored.validMoves().size());
    for (const Stockfish::Move move : mirrored.validMoves()) {
        mirroredActionIds.push_back(ChessEncoding::actionId(ChessAction(move), mirrored));
    }
    std::ranges::sort(mirroredActionIds);
    require(original.validMoves().size() == mirrored.validMoves().size(),
            "mirrored positions produced different legal-action counts");
    for (const Stockfish::Move move : original.validMoves()) {
        const int mirroredActionId =
            ChessEncoding::mirrorActionId(ChessEncoding::actionId(ChessAction(move), original));
        require(std::ranges::binary_search(mirroredActionIds, mirroredActionId),
                "policy-plane mirror did not map to the mirrored legal action set");
    }
}

void testDecodeRejections() {
    const Board initial;
    requireInvalidArgument(
        [&initial] { static_cast<void>(ChessEncoding::decodeAction(-1, initial)); },
        "negative chess action id was accepted");
    requireInvalidArgument(
        [&initial] {
            static_cast<void>(ChessEncoding::decodeAction(ChessEncoding::actionCount, initial));
        },
        "out-of-range chess action id was accepted");
    if constexpr (chessActionEncoding == ChessActionEncoding::PolicyPlane) {
        requireInvalidArgument(
            [&initial] { static_cast<void>(ChessEncoding::decodeAction(63, initial)); },
            "off-board chess action id was accepted");
    }
    requireInvalidArgument(
        [&initial] { static_cast<void>(ChessEncoding::decodeAction(0, initial)); },
        "position-illegal chess action id was accepted");
}

void testStableLegalOnlySoftmax() {
    const Board board;
    std::vector<float> policy(ChessEncoding::actionCount, std::numeric_limits<float>::quiet_NaN());
    for (const Stockfish::Move move : board.validMoves()) {
        policy[ChessEncoding::actionId(ChessAction(move), board)] = 1'000.0F;
    }
    SearchInferenceResult<ChessGame> uniform =
        processInferencePosition<ChessGame>(policy.data(), validOutcome.data(), board);
    require(uniform.actions.size() == board.validMoves().size(),
            "uniform fallback omitted legal actions");
    const float expected = 1.0F / static_cast<float>(uniform.actions.size());
    for (const auto &[action, actionId, probability] : uniform.actions) {
        static_cast<void>(action);
        static_cast<void>(actionId);
        require(std::abs(probability - expected) <= scoreTolerance,
                "legal-only softmax returned a non-uniform probability");
    }

    for (const Stockfish::Move move : board.validMoves()) {
        policy[ChessEncoding::actionId(ChessAction(move), board)] = 0.0F;
    }
    const int preferredAction =
        ChessEncoding::actionId(ChessAction(board.validMoves().front()), board);
    policy[preferredAction] += std::log(2.0F);
    const SearchInferenceResult<ChessGame> weighted =
        processInferencePosition<ChessGame>(policy.data(), validOutcome.data(), board);
    require(weighted.actions.size() == board.validMoves().size(), "softmax omitted legal actions");
    const auto preferred = std::ranges::find_if(
        weighted.actions, [&](const auto &entry) { return entry.action_id == preferredAction; });
    require(preferred != weighted.actions.end(), "softmax omitted the preferred legal action");
    require(std::abs(preferred->prior - 2.0F / 21.0F) <= scoreTolerance,
            "softmax did not preserve the requested legal-logit ratio");
}

void testGoPointPassLegalOnlySoftmax() {
    const GoRules rules{.komi_half_points = 15, .maximum_moves = 200};
    const Go7Game::State position(rules);
    const Go7Game::State occupied = position.child(GoAction<7>(0));
    std::vector<float> policy(GoRepresentationDimensions<7>::actionCount,
                              std::numeric_limits<float>::quiet_NaN());
    for (const GoAction<7> action : Go7Game::legalActions(occupied)) {
        policy[Go7Game::Encoding::actionId(action, occupied)] = 0.0F;
    }
    policy[GoAction<7>::passId] += std::log(3.0F);

    const SearchInferenceResult<Go7Game> result =
        processInferencePosition<Go7Game>(policy.data(), validOutcome.data(), occupied);
    require(result.actions.size() == 49, "Go legal-only softmax returned the wrong action count");
    require(result.actions.back().action.isPass(), "Go pass was not the final point-pass action");
    require(std::abs(result.actions.back().prior - 1.0F / 17.0F) <= scoreTolerance,
            "Go legal-only softmax did not preserve the pass-logit ratio");
    float probabilitySum = 0.0F;
    for (const auto &[action, actionId, probability] : result.actions) {
        static_cast<void>(actionId);
        require(action.id != 0, "Go legal-only softmax returned an occupied point");
        probabilitySum += probability;
    }
    require(std::abs(probabilitySum - 1.0F) <= scoreTolerance,
            "Go point-pass policy was not normalized over legal actions");
}

void testTerminalAndOutcomeValidation() {
    const Board terminal("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
    const std::vector<float> policy = finiteLogits();
    require(processInferencePosition<ChessGame>(policy.data(), validOutcome.data(), terminal)
                .actions.empty(),
            "terminal position returned policy actions");

    const std::array<std::array<float, 3>, 4> invalidOutcomes = {
        std::array<float, 3>{NAN, 0.5F, 0.5F},
        std::array<float, 3>{0.5F, INFINITY, 0.5F},
        std::array<float, 3>{-0.1F, 0.6F, 0.5F},
        std::array<float, 3>{0.2F, 0.2F, 0.2F},
    };
    for (const auto &outcome : invalidOutcomes) {
        bool rejected = false;
        try {
            static_cast<void>(
                processInferencePosition<ChessGame>(policy.data(), outcome.data(), Board{}));
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        require(rejected, "invalid WDL output was accepted");
    }
}
} // namespace

int runMovePolicyProcessingTests() {
    Stockfish::Bitboards::init();
    Stockfish::Position::init();
    testRepresentativePositions();
    testSelectedActionSpaceLayout();
    testReducedActionSpaceIsTotal();
    testPlayoutRoundTrips();
    testSpecialMoveRoundTrips();
    testSemanticMirroring();
    testDecodeRejections();
    testStableLegalOnlySoftmax();
    testGoPointPassLegalOnlySoftmax();
    testTerminalAndOutcomeValidation();
    std::cout << "Move policy processing tests passed\n";
    return 0;
}
