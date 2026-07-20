#include "MoveEncoding.hpp"

#include "bitboard.h"
#include "position.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr float SCORE_TOLERANCE = 1e-6f;

void require(const bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<EncodedMoveScore> torchReference(const torch::Tensor &policy, const Board &board) {
    const std::vector<Move> &legalMoves = board.validMoves();
    if (legalMoves.empty()) {
        return {};
    }

    torch::Tensor legalMoveMask = torch::zeros({ACTION_SIZE}, torch::kFloat32);
    for (const Move move : legalMoves) {
        legalMoveMask[encodeMove(move, &board)] = 1.0f;
    }

    torch::Tensor filteredPolicy = policy * legalMoveMask;
    const float legalPolicySum = filteredPolicy.sum().item<float>();
    if (legalPolicySum < 0.00001f) {
        filteredPolicy = legalMoveMask / legalMoveMask.sum();
    } else {
        filteredPolicy /= legalPolicySum;
    }

    const torch::Tensor nonzeroIndices = torch::nonzero(filteredPolicy > 0);
    std::vector<EncodedMoveScore> result;
    result.reserve(static_cast<size_t>(nonzeroIndices.size(0)));
    for (int64_t index = 0; index < nonzeroIndices.size(0); ++index) {
        const int encodedMove = nonzeroIndices[index].item<int>();
        result.emplace_back(encodedMove, filteredPolicy[encodedMove].item<float>());
    }
    return result;
}

void requireEquivalent(const Board &board, const torch::Tensor &policy,
                       const std::string &description) {
    const std::vector<EncodedMoveScore> expected = torchReference(policy, board);
    const std::vector<EncodedMoveScore> actual =
        filterPolicyThenGetMovesAndProbabilities(policy, &board);

    require(actual.size() == expected.size(), description + ": move count differs");
    for (size_t index = 0; index < expected.size(); ++index) {
        require(actual[index].first == expected[index].first,
                description + ": encoded move ordering differs");
        require(std::abs(actual[index].second - expected[index].second) <= SCORE_TOLERANCE,
                description + ": normalized score differs");
    }
}

torch::Tensor positivePolicy() {
    return torch::arange(1, ACTION_SIZE + 1, torch::TensorOptions().dtype(torch::kFloat32));
}

void testNormalPosition() {
    const Board board;
    requireEquivalent(board, positivePolicy(), "starting position");
}

void testBlackToMovePosition() {
    const Board board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
    requireEquivalent(board, positivePolicy(), "black-to-move position");
}

void testPromotionPosition() {
    const Board board("7k/P7/8/8/8/8/8/7K w - - 0 1");
    size_t promotionCount = 0;
    for (const Move move : board.validMoves()) {
        if (move.type_of() == PROMOTION) {
            ++promotionCount;
        }
    }
    require(promotionCount == 4, "promotion position did not contain four promotion choices");
    requireEquivalent(board, positivePolicy(), "promotion position");
}

void testUniformFallback() {
    const Board board;
    const torch::Tensor policy = torch::zeros({ACTION_SIZE}, torch::kFloat32);
    const std::vector<EncodedMoveScore> actual =
        filterPolicyThenGetMovesAndProbabilities(policy, &board);

    require(actual.size() == board.validMoves().size(), "uniform fallback omitted legal moves");
    const float expectedProbability = 1.0f / static_cast<float>(board.validMoves().size());
    for (const EncodedMoveScore &moveWithProbability : actual) {
        require(std::abs(moveWithProbability.second - expectedProbability) <= SCORE_TOLERANCE,
                "uniform fallback returned a non-uniform score");
    }
    requireEquivalent(board, policy, "uniform fallback");
}

void testZeroProbabilityMovesAreOmitted() {
    const Board board;
    torch::Tensor policy = torch::zeros({ACTION_SIZE}, torch::kFloat32);
    const std::vector<Move> &legalMoves = board.validMoves();
    policy[encodeMove(legalMoves[0], &board)] = 1.0f;
    policy[encodeMove(legalMoves[1], &board)] = 2.0f;

    const std::vector<EncodedMoveScore> actual =
        filterPolicyThenGetMovesAndProbabilities(policy, &board);
    require(actual.size() == 2, "zero-probability legal moves were retained");
    requireEquivalent(board, policy, "mixed zero and positive policy");
}

void testTerminalPosition() {
    const Board board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
    require(board.validMoves().empty(), "terminal test position has legal moves");
    require(filterPolicyThenGetMovesAndProbabilities(positivePolicy(), &board).empty(),
            "terminal position returned policy moves");
}
} // namespace

int main() {
    Bitboards::init();
    Position::init();

    testNormalPosition();
    testBlackToMovePosition();
    testPromotionPosition();
    testUniformFallback();
    testZeroProbabilityMovesAreOmitted();
    testTerminalPosition();
    std::cout << "Move policy processing tests passed\n";
    return 0;
}
