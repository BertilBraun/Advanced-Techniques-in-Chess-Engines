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
#include <set>
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

std::vector<float> positivePolicy() {
    std::vector<float> policy(ChessEncoding::action_count);
    for (const int actionId : range(ChessEncoding::action_count)) {
        policy[actionId] = static_cast<float>(actionId + 1);
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
        require(probability > 0.0F, description + ": retained non-positive probability");
        previousActionId = actionId;
        sum += probability;
    }
    require(std::abs(sum - 1.0F) <= scoreTolerance, description + ": policy is not normalized");
}

void testRepresentativePositions() {
    const Board initial;
    requireNormalized(initial, positivePolicy(), "starting position");
    requireNormalized(Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"),
                      positivePolicy(), "black-to-move position");
    const Board promotion("7k/P7/8/8/8/8/8/7K w - - 0 1");
    require(std::ranges::count_if(promotion.validMoves(),
                                  [](const Stockfish::Move move) {
                                      return move.type_of() == Stockfish::PROMOTION;
                                  }) == 4,
            "promotion position did not contain four promotion choices");
    requireNormalized(promotion, positivePolicy(), "promotion position");
}

void testSpatialPolicyMapping() {
    const ChessPolicyMapping &mapping = ChessEncoding::policyMapping();
    const auto &indices = mapping.action_plane_indices;
    const std::set<int> unique(indices.begin(), indices.end());
    require(unique.size() == ChessEncoding::action_count,
            "spatial policy mapping contains duplicate slots");
    require(*unique.begin() >= 0, "spatial policy mapping contains a negative slot");
    require(mapping.plane_count == ChessRepresentationDimensions::policy_plane_count,
            "spatial policy mapping returned an incorrect plane count");
    require(*unique.rbegin() < mapping.plane_count * 64,
            "spatial policy mapping exceeds its plane layout");

    const Board promotion("7k/P7/8/8/8/8/8/7K w - - 0 1");
    std::set<int> promotionPlanes;
    for (const Stockfish::Move move : promotion.validMoves()) {
        if (move.type_of() == Stockfish::PROMOTION) {
            const int actionId = ChessEncoding::actionId(ChessAction(move), promotion);
            promotionPlanes.insert(indices[actionId] / 64);
        }
    }
    require(promotionPlanes.size() == 4,
            "promotion choices do not occupy distinct spatial policy planes");
    require(*promotionPlanes.begin() >= 64,
            "promotion choices did not use explicit promotion planes");
}

void testUniformFallbackAndSparsePolicy() {
    const Board board;
    std::vector<float> policy(ChessEncoding::action_count, 0.0F);
    SearchInferenceResult<ChessGame> uniform =
        processInferencePosition<ChessGame>(policy.data(), validOutcome.data(), board);
    require(uniform.actions.size() == board.validMoves().size(),
            "uniform fallback omitted legal actions");
    const float expected = 1.0F / static_cast<float>(uniform.actions.size());
    for (const auto &[action, probability] : uniform.actions) {
        static_cast<void>(action);
        require(std::abs(probability - expected) <= scoreTolerance,
                "uniform fallback returned a non-uniform probability");
    }

    policy[ChessEncoding::actionId(ChessAction(board.validMoves()[0]), board)] = 1.0F;
    policy[ChessEncoding::actionId(ChessAction(board.validMoves()[1]), board)] = 2.0F;
    const SearchInferenceResult<ChessGame> sparse =
        processInferencePosition<ChessGame>(policy.data(), validOutcome.data(), board);
    require(sparse.actions.size() == 2, "zero-probability legal actions were retained");
}

void testTerminalAndOutcomeValidation() {
    const Board terminal("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
    const std::vector<float> policy = positivePolicy();
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
    testSpatialPolicyMapping();
    testUniformFallbackAndSparsePolicy();
    testTerminalAndOutcomeValidation();
    std::cout << "Move policy processing tests passed\n";
    return 0;
}
