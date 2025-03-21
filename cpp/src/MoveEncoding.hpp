#pragma once

#include "common.hpp"

using PolicyMove = std::pair<Move, float>;

inline std::vector<PolicyMove> filterPolicyThenGetMovesAndProbabilities(const torch::Tensor &policy,
                                                                        Board &board);

inline int encodeMove(const Move &move);

inline torch::Tensor encodeMoves(const std::vector<PolicyMove> &movesWithProbabilities,
                                 bool normalize = true);

inline Move decodeMove(int moveIndex);

inline std::vector<Move> decodeMoves(const std::vector<int> &moveIndices);

inline torch::Tensor flipActionProbabilitiesHorizontal(const torch::Tensor &actionProbabilities);
inline torch::Tensor flipActionProbabilitiesVertical(const torch::Tensor &actionProbabilities);

inline int flipMoveIndexHorizontal(int moveIndex);

inline int flipMoveIndexVertical(int moveIndex);

using MoveMapping = std::array<std::array<std::array<int, PieceType::NUM_PIECE_TYPES>, 64>, 64>;

inline std::pair<int, int> square_to_index(int square) { return {square / 8, square % 8}; }

namespace defines {
const std::vector<std::pair<int, int>> DIRECTIONS = {{1, 0},  {1, 1},   {0, 1},  {-1, 1},
                                                     {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
const std::vector<std::pair<int, int>> KNIGHT_MOVES = {{2, 1},   {1, 2},   {-1, 2}, {-2, 1},
                                                       {-2, -1}, {-1, -2}, {1, -2}, {2, -1}};
const std::vector<PieceType> PROMOTION_PIECES = {PieceType::QUEEN, PieceType::ROOK,
                                                 PieceType::BISHOP, PieceType::KNIGHT};
} // namespace defines

inline std::pair<MoveMapping, int> __precalculateMoveMappings() {
    MoveMapping moveMappings{};
    // fill with -1
    for (auto &fromSquare : moveMappings) {
        for (auto &toSquare : fromSquare) {
            for (auto &promotionType : toSquare) {
                promotionType = -1;
            }
        }
    }

    int index = 0;

    auto addMove = [&](int fromSquare, int toSquare, PieceType promotionType) {
        moveMappings[fromSquare][toSquare][(int) promotionType] = index++;
    };

    for (const auto from_square : range(64)) {
        const auto [row, col] = square_to_index(from_square);

        for (const auto& [dr, dc] : defines::DIRECTIONS) {
            for (int distance : range(1, 8)) {
                int toRow = row + dr * distance;
                int toCol = col + dc * distance;
                if (0 <= toRow && toRow < 8 && 0 <= toCol && toCol < 8) {
                    addMove(from_square, square(toCol, toRow), PieceType::NONE);
                }
            }
        }

        for (const auto &[dx, dy] : defines::KNIGHT_MOVES) {
            int toRow = row + dx;
            int toCol = col + dy;
            if (0 <= toRow && toRow < 8 && 0 <= toCol && toCol < 8) {
                addMove(from_square, square(toCol, toRow), PieceType::NONE);
            }
        }

        // Calculate pawn promotion moves from this square
        if (row == 1 || row == 6) {
            int toRow = row == 1 ? 0 : 7;

            for (int offset : {-1, 0, 1}) {
                if (0 <= col + offset && col + offset < 8) {
                    int toSquare = square(col + offset, toRow);
                    for (PieceType promotionType : defines::PROMOTION_PIECES) {
                        addMove(from_square, toSquare, promotionType);
                    }
                }
            }
        }
    }

    return {moveMappings, index};
}

inline std::array<std::tuple<Square, Square, PieceType>, ACTION_SIZE>
__precalculateReverseMoveMappings(const MoveMapping &moveMappings) {

    std::array<std::tuple<Square, Square, PieceType>, ACTION_SIZE> reverseMoveMappings;

    for (Square fromSquare : SQUARES) {
        for (Square toSquare : SQUARES) {
            for (PieceType promotionType : PIECE_TYPES_AND_NONE) {
                int moveIndex = moveMappings[fromSquare][toSquare][(int) promotionType];
                if (moveIndex != -1) {
                    reverseMoveMappings[moveIndex] = {fromSquare, toSquare, promotionType};
                }
            }
        }
    }

    return reverseMoveMappings;
}

inline std::pair<std::vector<int>, std::vector<int>>
__precalculateFlippedIndices(const MoveMapping &moveMappings) {

    std::vector<int> flippedIndicesHorizontal(ACTION_SIZE, -1);
    std::vector<int> flippedIndicesVertical(ACTION_SIZE, -1);

    for (Square fromSquare : SQUARES) {
        for (Square toSquare : SQUARES) {
            for (PieceType promotionType : PIECE_TYPES_AND_NONE) {
                int flippedFromHorizontal = squareFlipHorizontal(fromSquare);
                int flippedToHorizontal = squareFlipHorizontal(toSquare);
                int flippedFromVertical = squareFlipVertical(fromSquare);
                int flippedToVertical = squareFlipVertical(toSquare);

                int flippedMoveIndexHorizontal =
                    moveMappings[flippedFromHorizontal][flippedToHorizontal][(int) promotionType];
                int flippedMoveIndexVertical =
                    moveMappings[flippedFromVertical][flippedToVertical][(int) promotionType];
                int moveIndex = moveMappings[fromSquare][toSquare][(int) promotionType];

                if (moveIndex != -1) {
                    flippedIndicesHorizontal[moveIndex] = flippedMoveIndexHorizontal;
                    flippedIndicesVertical[moveIndex] = flippedMoveIndexVertical;
                }
            }
        }
    }

    return {flippedIndicesHorizontal, flippedIndicesVertical};
}

inline const auto __MOVE_MAPPINGS = __precalculateMoveMappings().first;
inline const auto __REVERSE_MOVE_MAPPINGS = __precalculateReverseMoveMappings(__MOVE_MAPPINGS);
inline const auto __FLIPPED_INDICES_HORIZONTAL =
    __precalculateFlippedIndices(__MOVE_MAPPINGS).first;
inline const auto __FLIPPED_INDICES_VERTICAL = __precalculateFlippedIndices(__MOVE_MAPPINGS).second;

inline torch::Tensor encodeMoves(const std::vector<PolicyMove> &movesWithProbabilities,
                                 bool normalize) {
    // Encodes a list of moves with their corresponding probabilities into a 1D tensor.
    //
    // The list of moves is a list of tuples, where each tuple contains a move and its
    // corresponding probability. The tensor has a length of TOTAL_MOVES, representing all
    // possible moves from all squares to all reachable squares. Each entry in the tensor
    // represents a possible move on the board. If the corresponding move is in the list of

    // moves, the entry is the probability of the move, and 0 otherwise.
    //
    // :param movesWithProbabilities: The list of moves with their corresponding probabilities.
    // :return: A 1D tensor representing the encoded moves.

    torch::Tensor movesEncoded;

    if (normalize) {
        // Initialize the tensor with -inf so that the softmax function will set the probability of
        // illegal moves to 0
        movesEncoded =
            torch::full({ACTION_SIZE}, -std::numeric_limits<float>::infinity(), torch::kFloat16);
    } else {
        movesEncoded = torch::zeros({ACTION_SIZE}, torch::kFloat16);
    }

    for (const auto &[move, probability] : movesWithProbabilities) {
        movesEncoded[encodeMove(move)] = probability;
    }

    if (!normalize) {
        return movesEncoded;
    }

    // Normalize the tensor to be a probability distribution
    // Softmax multiplied by 20 to make the probabilities more confident
    return torch::softmax(movesEncoded * 20.f, 0);
}

inline torch::Tensor __encodeLegalMoves(Board &board) {
    // Encodes the legal moves of a chess board into a 1D tensor.
    //
    // Each entry in the array represents a possible move on the board. If the
    // corresponding move is legal, the entry is 1, and 0 otherwise. The array
    // has a length of TOTAL_MOVES, representing all possible moves from all squares
    // to all reachable squares.
    //
    // :param board: The chess board to encode.
    // :return: A 1D tensor representing the encoded legal moves.

    std::vector<PolicyMove> movesWithProbabilities;
    for (const auto &move : board.legalMoves()) {
        movesWithProbabilities.emplace_back(move, 1.0);
    }
    return encodeMoves(movesWithProbabilities, false);
}

inline torch::Tensor __filterPolicyWithLegalMoves(const torch::Tensor &policy, Board &board) {
    // Filters a policy with the legal moves of a chess board.
    //
    // The policy is a 1D tensor representing the probabilities of each move
    // in the board. The legal moves are encoded in a 1D tensor, where each
    // entry is 1 if the corresponding move is legal, and 0 otherwise. The policy
    // is then filtered to only include the probabilities of the legal moves.
    //
    // :param policy: The policy to filter.
    // :param board: The chess board to filter the policy with.
    // :return: The filtered policy.

    torch::Tensor legalMovesEncoded = __encodeLegalMoves(board);
    torch::Tensor filteredPolicy = policy * legalMovesEncoded;
    filteredPolicy /= filteredPolicy.sum();
    return filteredPolicy;
}

inline std::vector<PolicyMove> __mapPolicyToMoves(const torch::Tensor &policy) {
    std::vector<PolicyMove> movesWithProbabilities;

    torch::Tensor nonzeroIndices = torch::nonzero(policy > 0);
    for (int i = 0; i < (int) nonzeroIndices.size(0); ++i) {
        int index = nonzeroIndices[i].item<int>();
        float probability = policy[index].item<float>();
        Move move = decodeMove(index);
        movesWithProbabilities.emplace_back(move, probability);
    }

    return movesWithProbabilities;
}

inline std::vector<PolicyMove> filterPolicyThenGetMovesAndProbabilities(const torch::Tensor &policy,
                                                                        Board &board) {
    // Gets a list of moves with their corresponding probabilities from a policy.
    //
    // The policy is a 1D tensor representing the probabilities of each move
    // in the board. The list of moves is a list of tuples, where each tuple contains
    // a move and its corresponding probability.
    //
    // :param policy: The policy to get the moves and probabilities from.
    // :param board: The chess board to filter the policy with.
    // :return: The list of moves with their corresponding probabilities.

    auto filteredPolicy = __filterPolicyWithLegalMoves(policy, board);
    auto movesWithProbabilities = __mapPolicyToMoves(filteredPolicy);
    return movesWithProbabilities;
}

inline int encodeMove(const Move &move) {
    // Encodes a chess move into a move index.
    //
    // :param move: The move to encode.
    // :return: The encoded move index.

    int moveIndex = __MOVE_MAPPINGS[move.fromSquare()][move.toSquare()][(int) move.promotion()];

    return moveIndex;
}

inline Move decodeMove(int moveIndex) {
    // Decodes a move index into a chess move.
    //
    // :param move_index: The index of the move to decode.
    // :return: The decoded chess move.

    auto [from_square, to_square, promotion_type] = __REVERSE_MOVE_MAPPINGS[moveIndex];
    return Move(from_square, to_square, promotion_type);
}

inline std::vector<Move> decodeMoves(const std::vector<int> &moveIndices) {
    // Decodes an array of move indices into a list of chess moves.
    //
    // :param moveIndices: The array of move indices to decode.
    // :return: The list of decoded chess moves.

    std::vector<Move> moves;
    moves.reserve(moveIndices.size());
    for (int moveIndex : moveIndices) {
        moves.emplace_back(decodeMove(moveIndex));
    }
    return moves;
}

inline torch::Tensor flipActionProbabilitiesHorizontal(const torch::Tensor &actionProbabilities) {
    torch::Tensor flippedProbabilities = torch::zeros_like(actionProbabilities);

    for (int idx = 0; idx < (int) actionProbabilities.size(0); ++idx) {
        int flippedIdx = flipMoveIndexHorizontal(idx);
        if (flippedIdx != -1) {
            flippedProbabilities[flippedIdx] = actionProbabilities[idx];
        }
    }

    return flippedProbabilities;
}

inline torch::Tensor flipActionProbabilitiesVertical(const torch::Tensor &actionProbabilities) {
    torch::Tensor flippedProbabilities = torch::zeros_like(actionProbabilities);

    for (int idx = 0; idx < (int) actionProbabilities.size(0); ++idx) {
        int flippedIdx = flipMoveIndexVertical(idx);
        if (flippedIdx != -1) {
            flippedProbabilities[flippedIdx] = actionProbabilities[idx];
        }
    }

    return flippedProbabilities;
}

inline int flipMoveIndexHorizontal(int moveIndex) {
    if (__FLIPPED_INDICES_HORIZONTAL[moveIndex] == -1) {
        log("Flipped index not found for", moveIndex, "in flipMoveIndexHorizontal");
    }
    return __FLIPPED_INDICES_HORIZONTAL[moveIndex];
}

inline int flipMoveIndexVertical(int moveIndex) {
    if (__FLIPPED_INDICES_VERTICAL[moveIndex] == -1) {
        log("Flipped index not found for", moveIndex, "in flipMoveIndexVertical");
    }
    return __FLIPPED_INDICES_VERTICAL[moveIndex];
}
