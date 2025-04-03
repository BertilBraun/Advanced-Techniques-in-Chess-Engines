#include "MoveEncoding.hpp"

namespace defines {
const std::vector<std::pair<int, int>> DIRECTIONS = {{1, 0},  {1, 1},   {0, 1},  {-1, 1},
                                                     {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
const std::vector<std::pair<int, int>> KNIGHT_MOVES = {{2, 1},   {1, 2},   {-1, 2}, {-2, 1},
                                                       {-2, -1}, {-1, -2}, {1, -2}, {2, -1}};
const std::vector<PieceType> PROMOTION_PIECES = {PieceType::QUEEN, PieceType::ROOK,
                                                 PieceType::BISHOP, PieceType::KNIGHT};
} // namespace defines

std::pair<MoveMapping, int> __precalculateMoveMappings() {
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

    for (const auto from_square : range(BOARD_SIZE)) {
        const auto [row, col] = squareToIndex(from_square);

        for (const auto &[dr, dc] : defines::DIRECTIONS) {
            for (int distance : range(1, BOARD_LENGTH)) {
                int toRow = row + dr * distance;
                int toCol = col + dc * distance;
                if (0 <= toRow && toRow < BOARD_LENGTH && 0 <= toCol && toCol < BOARD_LENGTH) {
                    addMove(from_square, square(toCol, toRow), PieceType::NONE);
                }
            }
        }

        for (const auto &[dx, dy] : defines::KNIGHT_MOVES) {
            int toRow = row + dx;
            int toCol = col + dy;
            if (0 <= toRow && toRow < BOARD_LENGTH && 0 <= toCol && toCol < BOARD_LENGTH) {
                addMove(from_square, square(toCol, toRow), PieceType::NONE);
            }
        }

        // Calculate pawn promotion moves from this square
        if (row == 1 || row == 6) {
            int toRow = row == 1 ? 0 : 7;

            for (int offset : {-1, 0, 1}) {
                if (0 <= col + offset && col + offset < BOARD_LENGTH) {
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

std::array<std::tuple<Square, Square, PieceType>, ACTION_SIZE>
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

const auto __MOVE_MAPPINGS = __precalculateMoveMappings().first;
const auto __REVERSE_MOVE_MAPPINGS = __precalculateReverseMoveMappings(__MOVE_MAPPINGS);

int encodeMove(const Move &move) {
    // Encodes a chess move into a move index.
    //
    // :param move: The move to encode.
    // :return: The encoded move index.

    int moveIndex = __MOVE_MAPPINGS[move.fromSquare()][move.toSquare()][(int) move.promotion()];

    return moveIndex;
}

Move decodeMove(int moveIndex) {
    // Decodes a move index into a chess move.
    //
    // :param move_index: The index of the move to decode.
    // :return: The decoded chess move.

    auto [from_square, to_square, promotion_type] = __REVERSE_MOVE_MAPPINGS[moveIndex];
    return Move(from_square, to_square, promotion_type);
}

torch::Tensor encodeMoves(const std::vector<Move> &moves) {
    // Encodes a list of moves into a 1D tensor.
    //
    // Each entry in the array represents a possible move on the board. If the
    // corresponding move is in the list, the entry is 1, and 0 otherwise. The array
    // has a length of TOTAL_MOVES, representing all possible moves from all squares
    // to all reachable squares.
    //
    // :param moves: The list of moves to encode.
    // :return: A 1D tensor representing the encoded moves.

    torch::Tensor movesEncoded = torch::zeros({ACTION_SIZE}, torch::kInt8);

    for (const Move &move : moves) {
        movesEncoded[encodeMove(move)] = 1;
    }

    return movesEncoded;
}

std::vector<Move> decodeMoves(const std::vector<int> &moveIndices) {
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

torch::Tensor __encodeLegalMoves(Board &board) {
    // Encodes the legal moves of a chess board into a 1D tensor.
    //
    // Each entry in the array represents a possible move on the board. If the
    // corresponding move is legal, the entry is 1, and 0 otherwise. The array
    // has a length of TOTAL_MOVES, representing all possible moves from all squares
    // to all reachable squares.
    //
    // :param board: The chess board to encode.
    // :return: A 1D tensor representing the encoded legal moves.
    return encodeMoves(board.legalMoves());
}

torch::Tensor __filterPolicyWithLegalMoves(const torch::Tensor &policy, Board &board) {
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

std::vector<MoveScore> __mapPolicyToMoves(const torch::Tensor &policy) {
    std::vector<MoveScore> movesWithProbabilities;

    torch::Tensor nonzeroIndices = torch::nonzero(policy > 0);
    for (int i = 0; i < (int) nonzeroIndices.size(0); ++i) {
        int index = nonzeroIndices[i].item<int>();
        float probability = policy[index].item<float>();
        movesWithProbabilities.emplace_back(index, probability);
    }

    return movesWithProbabilities;
}

std::vector<MoveScore> filterPolicyThenGetMovesAndProbabilities(const torch::Tensor &policy,
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

std::vector<MoveScore> filterMovesWithLegalMoves(const std::vector<MoveScore> &moves,
                                                 Board &board) {
    // Filters a list of moves with the legal moves of a chess board.
    //
    // The list of moves is a list of tuples, where each tuple contains a move
    // and its corresponding probability. The legal moves are encoded in a 1D tensor,
    // where each entry is 1 if the corresponding move is legal, and 0 otherwise.
    // The list of moves is then filtered to only include the legal moves.

    std::unordered_set<int> legalMovesSet;
    for (const Move &move : board.legalMoves()) {
        legalMovesSet.insert(encodeMove(move));
    }

    std::vector<MoveScore> filteredMoves;
    filteredMoves.reserve(moves.size());
    for (const auto &[move, probability] : moves) {
        if (legalMovesSet.find(move) != legalMovesSet.end()) {
            filteredMoves.emplace_back(move, probability);
        }
    }
    return filteredMoves;
}

torch::Tensor __filterPolicyWithLegalMovesAndEnPassantMoves(const torch::Tensor &policy,
                                                            Board &board) {
    // Filters a policy with the legal moves of a chess board but also allows all en passant moves.

    // The policy is a 1D tensor representing the probabilities of each move
    // in the board. The legal moves are encoded in a 1D tensor, where each
    // entry is 1 if the corresponding move is legal, and 0 otherwise. The policy
    // is then filtered to only include the probabilities of the legal moves.

    std::vector<Move> enPassantMoves = {
        // White en passant moves
        Move(A5, B6),
        Move(B5, A6),
        Move(B5, C6),
        Move(C5, B6),
        Move(C5, D6),
        Move(D5, C6),
        Move(D5, E6),
        Move(E5, D6),
        Move(E5, F6),
        Move(F5, E6),
        Move(F5, G6),
        Move(G5, F6),
        Move(G5, H6),
        Move(H5, G6),
        // Black en passant moves
        Move(A4, B3),
        Move(B4, A3),
        Move(B4, C3),
        Move(C4, B3),
        Move(C4, D3),
        Move(D4, C3),
        Move(D4, E3),
        Move(E4, D3),
        Move(E4, F3),
        Move(F4, E3),
        Move(F4, G3),
        Move(G4, F3),
        Move(G4, H3),
        Move(H4, G3),
    };

    auto allMoves = board.legalMoves();
    extend(allMoves, enPassantMoves);

    torch::Tensor legalMovesEncoded = encodeMoves(allMoves);
    torch::Tensor filteredPolicy = policy * legalMovesEncoded;
    float policySum = filteredPolicy.sum().item<float>();
    if (policySum == 0) {
        filteredPolicy = legalMovesEncoded / legalMovesEncoded.sum();
    } else {
        filteredPolicy /= policySum;
    }
    return filteredPolicy;
}

std::vector<MoveScore>
filterPolicyWithEnPassantMovesThenGetMovesAndProbabilities(const torch::Tensor &policy,
                                                           Board &board) {
    // Gets a list of moves with their corresponding probabilities from a policy.

    // The policy is a 1D tensor representing the probabilities of each move
    // in the board. The list of moves is a list of tuples, where each tuple contains
    // a move and its corresponding probability.

    auto filteredPolicy = __filterPolicyWithLegalMovesAndEnPassantMoves(policy, board);
    auto movesWithProbabilities = __mapPolicyToMoves(filteredPolicy);
    return movesWithProbabilities;
}
