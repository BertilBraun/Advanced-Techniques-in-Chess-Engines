#include "games/chess/ChessGameContract.hpp"
#include "search/SearchInference.hpp"

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

std::vector<float> positivePolicy() {
    std::vector<float> policy(ChessAction::action_count);
    for (int actionId = 0; actionId < ChessAction::action_count; ++actionId) {
        policy[actionId] = static_cast<float>(actionId + 1);
    }
    return policy;
}

void requireNormalized(const Board &board, const std::vector<float> &policy,
                       const std::string &description) {
    const SearchInferenceResult<ChessGameContract> result =
        processSearchInference<ChessGameContract>(policy.data(), validOutcome.data(), board);
    float sum = 0.0F;
    int previousActionId = -1;
    for (const auto &[action, probability] : result.actions) {
        const int actionId = ChessGameContract::actionId(action, board);
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

void testUniformFallbackAndSparsePolicy() {
    const Board board;
    std::vector<float> policy(ChessAction::action_count, 0.0F);
    SearchInferenceResult<ChessGameContract> uniform =
        processSearchInference<ChessGameContract>(policy.data(), validOutcome.data(), board);
    require(uniform.actions.size() == board.validMoves().size(),
            "uniform fallback omitted legal actions");
    const float expected = 1.0F / static_cast<float>(uniform.actions.size());
    for (const auto &[action, probability] : uniform.actions) {
        static_cast<void>(action);
        require(std::abs(probability - expected) <= scoreTolerance,
                "uniform fallback returned a non-uniform probability");
    }

    policy[ChessActionCodec::encode(ChessAction(board.validMoves()[0]), board)] = 1.0F;
    policy[ChessActionCodec::encode(ChessAction(board.validMoves()[1]), board)] = 2.0F;
    const SearchInferenceResult<ChessGameContract> sparse =
        processSearchInference<ChessGameContract>(policy.data(), validOutcome.data(), board);
    require(sparse.actions.size() == 2, "zero-probability legal actions were retained");
}

void testTerminalAndOutcomeValidation() {
    const Board terminal("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
    const std::vector<float> policy = positivePolicy();
    require(processSearchInference<ChessGameContract>(policy.data(), validOutcome.data(), terminal)
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
                processSearchInference<ChessGameContract>(policy.data(), outcome.data(), Board{}));
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        require(rejected, "invalid WDL output was accepted");
    }
}
} // namespace

int main() {
    Stockfish::Bitboards::init();
    Stockfish::Position::init();
    testRepresentativePositions();
    testUniformFallbackAndSparsePolicy();
    testTerminalAndOutcomeValidation();
    std::cout << "Move policy processing tests passed\n";
    return 0;
}
