#pragma once

#include "common.hpp"

using PolicyMove = std::pair<Move, float>;

inline std::vector<PolicyMove> filterPolicyThenGetMovesAndProbabilities(const torch::Tensor &policy,
                                                                        Board &board);

inline int encodeMove(const Move &move, Color currentColor);

inline torch::Tensor encodeMoves(const std::vector<PolicyMove> &movesWithProbabilities,
                                 Color currentColor, bool normalize = true);

inline Move decodeMove(int moveIndex, Color currentColor);

inline std::vector<Move> decodeMoves(const std::vector<int> &moveIndices, Color currentColor);

inline torch::Tensor flipActionProbabilitiesHorizontal(const torch::Tensor &actionProbabilities);
inline torch::Tensor flipActionProbabilitiesVertical(const torch::Tensor &actionProbabilities);

inline int flipMoveIndexHorizontal(int moveIndex);

inline int flipMoveIndexVertical(int moveIndex);

using MoveMapping = std::array<std::array<std::array<int, PieceType::NUM_PIECE_TYPES>, 64>, 64>;

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

    auto addPromotionMoves = [&](int fromSquare, int col, int toRow) {
        for (int offset : {-1, 0, 1}) {
            if (0 <= col + offset && col + offset < 8) {
                int toSquare = square(col + offset, toRow);
                addMove(fromSquare, toSquare, PieceType::QUEEN);
                addMove(fromSquare, toSquare, PieceType::ROOK);
                addMove(fromSquare, toSquare, PieceType::BISHOP);
                addMove(fromSquare, toSquare, PieceType::KNIGHT);
            }
        }
    };

    for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 8; ++col) {
            int fromSquare = square(col, row);

            for (const auto &[dx, dy] : KNIGHT_MOVES) {
                if (0 <= row + dx && row + dx < 8 && 0 <= col + dy && col + dy < 8) {
                    int toSquare = square(col + dy, row + dx);
                    addMove(fromSquare, toSquare, PieceType::NONE);
                }
            }

            for (const auto &[dx, dy] : ROOK_MOVES) {
                for (int i = 1; i < 8; ++i) {
                    if (0 <= row + i * dx && row + i * dx < 8 && 0 <= col + i * dy &&
                        col + i * dy < 8) {
                        int toSquare = square(col + i * dy, row + i * dx);
                        addMove(fromSquare, toSquare, PieceType::NONE);
                    }
                }
            }

            for (const auto &[dx, dy] : BISHOP_MOVES) {
                for (int i = 1; i < 8; ++i) {
                    if (0 <= row + i * dx && row + i * dx < 8 && 0 <= col + i * dy &&
                        col + i * dy < 8) {
                        int toSquare = square(col + i * dy, row + i * dx);
                        addMove(fromSquare, toSquare, PieceType::NONE);
                    }
                }
            }

            // Calculate pawn promotion moves from this square
            if (row == 1) {
                addPromotionMoves(fromSquare, col, row - 1);
            } else if (row == 6) {
                addPromotionMoves(fromSquare, col, row + 1);
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
                                 Color currentColor, bool normalize) {
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
        movesEncoded[encodeMove(move, currentColor)] = probability;
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
    return encodeMoves(movesWithProbabilities, board.turn, false);
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

inline std::vector<PolicyMove> __mapPolicyToMoves(const torch::Tensor &policy, Color currentColor) {
    std::vector<PolicyMove> movesWithProbabilities;

    torch::Tensor nonzeroIndices = torch::nonzero(policy > 0);
    for (int i = 0; i < (int) nonzeroIndices.size(0); ++i) {
        int index = nonzeroIndices[i].item<int>();
        float probability = policy[index].item<float>();
        Move move = decodeMove(index, currentColor);
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
    auto movesWithProbabilities = __mapPolicyToMoves(filteredPolicy, board.turn);
    return movesWithProbabilities;
}

inline int encodeMove(const Move &move, Color currentColor) {
    // Encodes a chess move into a move index.
    //
    // :param move: The move to encode.
    // :param current_player: The current player to encode the move for.
    // :return: The encoded move index.

    int moveIndex = __MOVE_MAPPINGS[move.fromSquare()][move.toSquare()][(int) move.promotion()];

    if (currentColor == Color::BLACK) {
        return flipMoveIndexVertical(moveIndex);
    }

    return moveIndex;
}

inline Move decodeMove(int moveIndex, Color currentColor) {
    // Decodes a move index into a chess move.
    //
    // :param move_index: The index of the move to decode.
    // :return: The decoded chess move.

    if (currentColor == Color::BLACK) {
        moveIndex = flipMoveIndexVertical(moveIndex);
    }

    auto [from_square, to_square, promotion_type] = __REVERSE_MOVE_MAPPINGS[moveIndex];
    return Move(from_square, to_square, promotion_type);
}

inline std::vector<Move> decodeMoves(const std::vector<int> &moveIndices, Color currentColor) {
    // Decodes an array of move indices into a list of chess moves.
    //
    // :param moveIndices: The array of move indices to decode.
    // :return: The list of decoded chess moves.

    std::vector<Move> moves;
    for (int moveIndex : moveIndices) {
        moves.push_back(decodeMove(moveIndex, currentColor));
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
