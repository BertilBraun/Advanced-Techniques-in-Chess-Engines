#include "TestRunner.hpp"
#include "games/chess/ChessGame.hpp"
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

std::vector<float> finiteLogits() {
    std::vector<float> policy(ChessEncoding::action_count);
    for (const int actionId : range(ChessEncoding::action_count)) {
        policy[actionId] = 1000.0F + static_cast<float>(actionId) /
                                         static_cast<float>(ChessEncoding::action_count);
    }
    return policy;
}

void requireNormalized(const Board &board, const std::vector<float> &policy,
                       const std::string &description) {
    const SearchInferenceResult<ChessGame> result =
        processInferencePosition<ChessGame>(policy.data(), validOutcome.data(), board);
    float sum = 0.0F;
    int previousActionId = -1;
    for (const auto &[action, probability] : result.actions) {
        const int actionId = ChessGame::Encoding::actionId(action, board);
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

void testDirectPolicyLayout() {
    require(ChessEncoding::action_count == ChessRepresentationDimensions::policy_plane_count * 64,
            "chess action space is not the direct policy-plane layout");
    for (const int actionId : range(ChessEncoding::action_count)) {
        require(ChessEncoding::mirrorActionId(ChessEncoding::mirrorActionId(actionId)) == actionId,
                "policy-plane mirroring is not an involution");
    }
    const Board promotion("7k/P7/8/8/8/8/8/7K w - - 0 1");
    std::vector<int> promotionPlanes;
    for (const Stockfish::Move move : promotion.validMoves()) {
        if (move.type_of() == Stockfish::PROMOTION) {
            const int actionId = ChessEncoding::actionId(ChessAction(move), promotion);
            require(actionId % 64 == static_cast<int>(move.from_sq()),
                    "white promotion action does not retain its canonical origin square");
            promotionPlanes.push_back(actionId / 64);
        }
    }
    std::ranges::sort(promotionPlanes);
    require(std::ranges::unique(promotionPlanes).begin() == promotionPlanes.end() &&
                promotionPlanes.size() == 4,
            "promotion choices do not occupy distinct spatial policy planes");
    require(promotionPlanes.front() >= 64,
            "promotion choices did not use explicit promotion planes");

    const Board blackPromotion("7K/8/8/8/8/8/p7/7k b - - 0 1");
    for (const Stockfish::Move move : blackPromotion.validMoves()) {
        if (move.type_of() == Stockfish::PROMOTION) {
            const int actionId = ChessEncoding::actionId(ChessAction(move), blackPromotion);
            require(actionId % 64 == static_cast<int>(Stockfish::flip_rank(move.from_sq())),
                    "black promotion action does not retain its canonical origin square");
            require(ChessEncoding::decodeAction(actionId, blackPromotion) == ChessAction(move),
                    "black promotion action did not round-trip through the direct layout");
        }
    }
}

void testStableLegalOnlySoftmax() {
    const Board board;
    std::vector<float> policy(ChessEncoding::action_count, std::numeric_limits<float>::quiet_NaN());
    for (const Stockfish::Move move : board.validMoves()) {
        policy[ChessEncoding::actionId(ChessAction(move), board)] = 1'000.0F;
    }
    SearchInferenceResult<ChessGame> uniform =
        processInferencePosition<ChessGame>(policy.data(), validOutcome.data(), board);
    require(uniform.actions.size() == board.validMoves().size(),
            "uniform fallback omitted legal actions");
    const float expected = 1.0F / static_cast<float>(uniform.actions.size());
    for (const auto &[action, probability] : uniform.actions) {
        static_cast<void>(action);
        require(std::abs(probability - expected) <= scoreTolerance,
                "legal-only softmax returned a non-uniform probability");
    }

    const int preferredAction =
        ChessEncoding::actionId(ChessAction(board.validMoves().front()), board);
    policy[preferredAction] += std::log(2.0F);
    const SearchInferenceResult<ChessGame> weighted =
        processInferencePosition<ChessGame>(policy.data(), validOutcome.data(), board);
    require(weighted.actions.size() == board.validMoves().size(), "softmax omitted legal actions");
    const auto preferred = std::ranges::find_if(weighted.actions, [&](const auto &entry) {
        return ChessEncoding::actionId(entry.first, board) == preferredAction;
    });
    require(preferred != weighted.actions.end(), "softmax omitted the preferred legal action");
    require(std::abs(preferred->second - 2.0F / 21.0F) <= scoreTolerance,
            "softmax did not preserve the requested legal-logit ratio");
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
    testDirectPolicyLayout();
    testStableLegalOnlySoftmax();
    testTerminalAndOutcomeValidation();
    std::cout << "Move policy processing tests passed\n";
    return 0;
}
