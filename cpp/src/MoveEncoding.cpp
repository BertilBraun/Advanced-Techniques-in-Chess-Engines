#include "MoveEncoding.hpp"

typedef std::array<std::array<std::array<int, PieceType::PIECE_TYPE_NB>, BOARD_SIZE>, BOARD_SIZE>
    MoveMapping;

namespace defines {
const std::vector<std::pair<int, int>> DIRECTIONS = {{1, 0},  {1, 1},   {0, 1},  {-1, 1},
                                                     {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
const std::vector<std::pair<int, int>> KNIGHT_MOVES = {{2, 1},   {1, 2},   {-1, 2}, {-2, 1},
                                                       {-2, -1}, {-1, -2}, {1, -2}, {2, -1}};
const std::vector<PieceType> PROMOTION_PIECES = {PieceType::QUEEN};
// Note: not relevant for strong amateur play: PieceType::ROOK, PieceType::BISHOP, PieceType::KNIGHT};
} // namespace defines


MoveMapping precalculateMoveMappings() {
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

    for (const auto fromSquare : range(BOARD_SIZE)) {
        const auto [row, col] = squareToIndex(fromSquare);

        for (const auto &[dr, dc] : defines::DIRECTIONS) {
            for (const int distance : range(1, BOARD_LENGTH)) {
                const int toRow = row + dr * distance;
                const int toCol = col + dc * distance;
                if (0 <= toRow && toRow < BOARD_LENGTH && 0 <= toCol && toCol < BOARD_LENGTH) {
                    addMove(fromSquare,  square(toCol, toRow), PieceType::NO_PIECE_TYPE);
                }
            }
        }

        for (const auto &[dx, dy] : defines::KNIGHT_MOVES) {
            const int toRow = row + dx;
            const int toCol = col + dy;
            if (0 <= toRow && toRow < BOARD_LENGTH && 0 <= toCol && toCol < BOARD_LENGTH) {
                addMove(fromSquare, square(toCol, toRow), PieceType::NO_PIECE_TYPE);
            }
        }

        // Calculate pawn promotion moves from this square
        // Note: we don't need blacks promotion moves anymore, as black moves are always mirrored to the equivalent white moves before
        if (row == 6) {
            for (const int offset : {-1, 0, 1}) {
                if (0 <= col + offset && col + offset < BOARD_LENGTH) {
                    const int toSquare = square(col + offset, 6 + 1);
                    for (const PieceType promotionType : defines::PROMOTION_PIECES) {
                        addMove(fromSquare, toSquare, promotionType);
                    }
                }
            }
        }
    }

    return moveMappings;
}

std::array<std::tuple<Square, Square, PieceType>, ACTION_SIZE>
precalculateReverseMoveMappings(const MoveMapping &moveMappings) {

    std::array<std::tuple<Square, Square, PieceType>, ACTION_SIZE> reverseMoveMappings;

    for (Square fromSquare : range(Square::SQUARE_NB)) {
        for (Square toSquare : range(Square::SQUARE_NB)) {
            for (PieceType promotionType : range(PieceType::PIECE_TYPE_NB)) {
                int moveIndex = moveMappings[fromSquare][toSquare][(int) promotionType];
                if (moveIndex != -1) {
                    reverseMoveMappings[moveIndex] = {fromSquare, toSquare, promotionType};
                }
            }
        }
    }

    return reverseMoveMappings;
}

const auto MOVE_MAPPINGS = precalculateMoveMappings();
const auto REVERSE_MOVE_MAPPINGS = precalculateReverseMoveMappings(MOVE_MAPPINGS);

int encodeMove(Move move, const Board& board) {
    // Encodes a chess move into a move index.
    //
    // :param move: The move to encode.
    // :return: The encoded move index.

    if (board.)

    int moveIndex = MOVE_MAPPINGS[move.from_sq()][move.to_sq()][(int) move.promotion_type()];

    return moveIndex;
}

Move decodeMove(int moveIndex, const Board& board) {
    // Decodes a move index into a chess move.
    //
    // :param move_index: The index of the move to decode.
    // :return: The decoded chess move.

    auto [from_square, to_square, promotion_type] = REVERSE_MOVE_MAPPINGS[moveIndex];

    // TODO warning, the flags have to be set manually, maybe directly when iterating through the valid moves, just taking the valid move from the move gen, if the squares and promotion matches
    return Move::make(from_square, to_square, promotion_type);
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
    if (moveIndices.empty()) {
        return {};
    }

    std::vector<Move> moves;
    moves.reserve(moveIndices.size());
    for (int moveIndex : moveIndices) {
        moves.emplace_back(decodeMove(moveIndex));
    }
    return moves;
}

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
    return encodeMoves(board.legalMoves());
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

std::vector<MoveScore> mapPolicyToMoves(const torch::Tensor &policy) {
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

    auto filteredPolicy = filterPolicyWithLegalMoves(policy, board);
    auto movesWithProbabilities = mapPolicyToMoves(filteredPolicy);
    return movesWithProbabilities;
}
