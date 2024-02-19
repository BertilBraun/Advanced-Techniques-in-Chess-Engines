#include "MoveEncoding.hpp"

#include "common.hpp"

using MoveMapping = std::array<std::array<std::array<int, PieceType::NUM_PIECE_TYPES>, 64>, 64>;

std::pair<MoveMapping, int> precalculateMoveMappings() {
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

std::array<std::tuple<int, int, PieceType>, ACTION_SIZE>
precalculateReverseMoveMappings(const MoveMapping &moveMappings) {

    std::array<std::tuple<int, int, PieceType>, ACTION_SIZE> reverseMoveMappings;

    for (Square fromSquare : SQUARES) {
        for (Square toSquare : SQUARES) {
            for (PieceType promotionType : PIECE_TYPES) {
                int moveIndex = moveMappings[fromSquare][toSquare][(int) promotionType];
                if (moveIndex != -1) {
                    reverseMoveMappings[moveIndex] = {fromSquare, toSquare, promotionType};
                }
            }
        }
    }

    return reverseMoveMappings;
}

std::pair<std::vector<int>, std::vector<int>>
precalculateFlippedIndices(const MoveMapping &moveMappings) {

    std::vector<int> flippedIndicesHorizontal(ACTION_SIZE, -1);
    std::vector<int> flippedIndicesVertical(ACTION_SIZE, -1);

    for (Square fromSquare : SQUARES) {
        for (Square toSquare : SQUARES) {
            for (PieceType promotionType : PIECE_TYPES) {
                int flippedFromHorizontal = flipSquareHorizontal(fromSquare);
                int flippedToHorizontal = flipSquareHorizontal(toSquare);
                int flippedFromVertical = flipSquareVertical(fromSquare);
                int flippedToVertical = flipSquareVertical(toSquare);

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

const auto __MOVE_MAPPINGS = precalculateMoveMappings().first;
const auto __REVERSE_MOVE_MAPPINGS = precalculateReverseMoveMappings(__MOVE_MAPPINGS);
const auto __FLIPPED_INDICES_HORIZONTAL = precalculateFlippedIndices(__MOVE_MAPPINGS).first;
const auto __FLIPPED_INDICES_VERTICAL = precalculateFlippedIndices(__MOVE_MAPPINGS).second;

torch::Tensor encodeLegalMoves(Board &board) {
    // Encodes the legal moves of a chess board into a 1D tensor.
    //
    // Each entry in the array represents a possible move on the board. If the
    // corresponding move is legal, the entry is 1, and 0 otherwise. The array
    // has a length of TOTAL_MOVES, representing all possible moves from all squares
    // to all reachable squares.
    //
    // :param board: The chess board to encode.
    // :return: A 1D tensor representing the encoded legal moves.

    auto legalMovesEncoded = torch::zeros({ACTION_SIZE}, torch::kFloat32);

    for (const auto &move : board.legalMoves()) {
        legalMovesEncoded[encodeMove(move)] = 1.0;
    }

    return legalMovesEncoded;
}

torch::Tensor filterPolicyWithLegalMoves(const torch::Tensor &policy, Board &board) {
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

    torch::Tensor legalMovesEncoded = encodeLegalMoves(board);
    torch::Tensor filteredPolicy = policy * legalMovesEncoded;
    filteredPolicy /= filteredPolicy.sum();
    return filteredPolicy;
}

std::vector<std::pair<Move, float>> mapPolicyToMoves(const torch::Tensor &policy) {
    std::vector<std::pair<Move, float>> movesWithProbabilities;

    torch::Tensor nonzeroIndices = torch::nonzero(policy > 0).squeeze();
    for (int i = 0; i < (int) nonzeroIndices.size(0); ++i) {
        int index = nonzeroIndices[i].item<int>();
        float probability = policy[index].item<float>();
        Move move = decodeMove(index);
        movesWithProbabilities.emplace_back(move, probability);
    }

    return movesWithProbabilities;
}

std::vector<std::pair<Move, float>>
filterPolicyThenGetMovesAndProbabilities(const torch::Tensor &policy, Board &board) {
    // Gets a list of moves with their corresponding probabilities from a policy.
    //
    // The policy is a 1D tensor representing the probabilities of each move
    // in the board. The list of moves is a list of tuples, where each tuple contains
    // a move and its corresponding probability.
    //
    // :param policy: The policy to get the moves and probabilities from.
    // :param board: The chess board to filter the policy with.
    // :return: The list of moves with their corresponding probabilities.

    auto filteredPolicy = filterPolicyWithLegalMoves(policy, board);
    auto movesWithProbabilities = mapPolicyToMoves(filteredPolicy);
    return movesWithProbabilities;
}

int encodeMove(const Move &move) {
    // Encodes a chess move into a move index.
    //
    // :param move: The move to encode.
    // :param current_player: The current player to encode the move for.
    // :return: The encoded move index.

    return __MOVE_MAPPINGS[move.fromSquare()][move.toSquare()][(int) move.promotion()];
}

Move decodeMove(int moveIndex) {
    // Decodes a move index into a chess move.
    //
    // :param move_index: The index of the move to decode.
    // :return: The decoded chess move.

    auto [from_square, to_square, promotion_type] = __REVERSE_MOVE_MAPPINGS[moveIndex];
    return Move(from_square, to_square, promotion_type);
}

std::vector<Move> decodeMoves(const std::vector<int> &moveIndices) {
    // Decodes an array of move indices into a list of chess moves.
    //
    // :param moveIndices: The array of move indices to decode.
    // :return: The list of decoded chess moves.

    std::vector<Move> moves;
    for (int moveIndex : moveIndices) {
        auto [fromSquare, toSquare, promotionType] = __REVERSE_MOVE_MAPPINGS[moveIndex];
        moves.emplace_back(fromSquare, toSquare, promotionType);
    }
    return moves;
}

Square flipSquareHorizontal(const Square &square) {
    // Flip the file of the square, keeping the rank constant
    int file = squareFile(square);
    int rank = squareRank(square);
    auto flippedFile = 7 - file; // 0 becomes 7, 1 becomes 6, ..., 7 becomes 0
    return (Square) (rank * 8 + flippedFile);
}

Square flipSquareVertical(const Square &square) {
    // Flip the rank of the square, keeping the file constant
    int file = squareFile(square);
    int rank = squareRank(square);
    auto flippedRank = 7 - rank; // 0 becomes 7, 1 becomes 6, ..., 7 becomes 0
    return (Square) (flippedRank * 8 + file);
}

torch::Tensor flipActionProbabilities(const torch::Tensor &actionProbabilities,
                                      const std::function<int(int)> &flipMoveIndex) {
    torch::Tensor flippedProbabilities = torch::zeros_like(actionProbabilities);

    for (int idx = 0; idx < (int) actionProbabilities.size(0); ++idx) {
        int flippedIdx = flipMoveIndex(idx);
        if (flippedIdx != -1) {
            flippedProbabilities[flippedIdx] = actionProbabilities[idx];
        }
    }

    return flippedProbabilities;
}

int flipMoveIndexHorizontal(int moveIndex) { return __FLIPPED_INDICES_HORIZONTAL[moveIndex]; }

int flipMoveIndexVertical(int moveIndex) { return __FLIPPED_INDICES_VERTICAL[moveIndex]; }
