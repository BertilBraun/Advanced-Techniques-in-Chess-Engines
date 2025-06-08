#include "MoveEncoding.hpp"

namespace defines {
const std::vector<std::pair<int, int>> DIRECTIONS = {{1, 0},  {1, 1},   {0, 1},  {-1, 1},
                                                     {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
const std::vector<std::pair<int, int>> KNIGHT_MOVES = {{2, 1},   {1, 2},   {-1, 2}, {-2, 1},
                                                       {-2, -1}, {-1, -2}, {1, -2}, {2, -1}};
constexpr std::array PROMOTION_PIECES = {PieceType::QUEEN};
// Note: not relevant for strong amateur play: PieceType::ROOK, PieceType::BISHOP,
// PieceType::KNIGHT};

constexpr int NUM_PROMOTION_PIECES = (int) PieceType::PIECE_TYPE_NB;
} // namespace defines

typedef std::array<std::array<std::array<int, defines::NUM_PROMOTION_PIECES>, BOARD_SIZE>,
                   BOARD_SIZE>
    MoveMapping;

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
                    addMove(fromSquare, square(toCol, toRow), PieceType::NO_PIECE_TYPE);
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
        // Note: we don't need blacks promotion moves anymore, as black moves are always mirrored to
        // the equivalent white moves before
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

    for (int fromSquare : range(BOARD_SIZE)) {
        for (int toSquare : range(BOARD_SIZE)) {
            for (int promotionType : range(defines::NUM_PROMOTION_PIECES)) {
                const int moveIndex = moveMappings[fromSquare][toSquare][promotionType];
                if (moveIndex != -1) {
                    reverseMoveMappings[moveIndex] = {
                        static_cast<Square>(fromSquare),
                        static_cast<Square>(toSquare),
                        static_cast<PieceType>(promotionType),
                    };
                }
            }
        }
    }

    return reverseMoveMappings;
}

const auto MOVE_MAPPINGS = precalculateMoveMappings();
const auto REVERSE_MOVE_MAPPINGS = precalculateReverseMoveMappings(MOVE_MAPPINGS);

int encodeMove(const Move move, const Board *board) {
    // Encodes a chess move into a move index.
    //
    // param move: The move to encode.
    // param board: The current board state.
    // :return: The encoded move index.

    Square fromSquare = move.from_sq();
    Square toSquare = move.to_sq();
    const PieceType promotionType = move.promotion_type();

    if (board->currentPlayer() == -1) {
        // Flip the move to match the board's perspective.
        fromSquare = flip_file(fromSquare);
        toSquare = flip_file(toSquare);
    }

    return MOVE_MAPPINGS[fromSquare][toSquare][promotionType];
}

torch::Tensor encodeMoves(const std::vector<Move> &moves, const Board *board) {
    // Encodes a list of moves into a 1D tensor.
    //
    // Each entry in the array represents a possible move on the board. If the
    // corresponding move is in the list, the entry is 1, and 0 otherwise. The array
    // has a length of TOTAL_MOVES, representing all possible moves from all squares
    // to all reachable squares.
    //
    // param moves: The list of moves to encode.
    // param board: The current board state.
    // :return: A 1D tensor representing the encoded moves.

    torch::Tensor movesEncoded = torch::zeros({ACTION_SIZE}, torch::kInt8);

    for (const Move &move : moves) {
        movesEncoded[encodeMove(move, board)] = 1;
    }

    return movesEncoded;
}

std::vector<Move> decodeMoves(const std::vector<int> &moveIndices, const Board *board) {
    // Decodes an array of move indices into a list of chess moves.
    //
    // param moveIndices: The array of move indices to decode.
    // param board: The current board state.
    // :return: The list of decoded chess moves.
    if (moveIndices.empty()) {
        return {};
    }

    const std::vector<Move> legalMoves = board->validMoves();

    std::vector<Move> moves;
    moves.reserve(moveIndices.size());
    for (const int moveIndex : moveIndices) {

        auto [from_square, to_square, promotion_type] = REVERSE_MOVE_MAPPINGS[moveIndex];

        if (board->currentPlayer() == -1) {
            from_square = flip_file(from_square);
            to_square = flip_file(to_square);
        }

        for (Move move : legalMoves) {
            if (move.from_sq() == from_square && move.to_sq() == to_square &&
                move.promotion_type() == promotion_type) {
                // If the move matches the from and to square and the promotion type, we can use it.
                moves.emplace_back(move);
                break; // No need to check further, we found the move.
            }
        }
    }
    return moves;
}

torch::Tensor encodeLegalMoves(const Board *board) {
    // Encodes the legal moves of a chess board into a 1D tensor.
    //
    // Each entry in the array represents a possible move on the board. If the
    // corresponding move is legal, the entry is 1, and 0 otherwise. The array
    // has a length of TOTAL_MOVES, representing all possible moves from all squares
    // to all reachable squares.
    //
    // param board: The chess board to encode.
    // :return: A 1D tensor representing the encoded legal moves.
    return encodeMoves(board->validMoves(), board);
}

torch::Tensor filterPolicyWithLegalMoves(const torch::Tensor &policy, const Board *board) {
    // Filters a policy with the legal moves of a chess board.
    //
    // The policy is a 1D tensor representing the probabilities of each move
    // in the board. The legal moves are encoded in a 1D tensor, where each
    // entry is 1 if the corresponding move is legal, and 0 otherwise. The policy
    // is then filtered to only include the probabilities of the legal moves.
    //
    // param policy: The policy to filter.
    // param board: The chess board to filter the policy with.
    // :return: The filtered policy.

    const torch::Tensor legalMovesEncoded = encodeLegalMoves(board);
    const torch::Tensor filteredPolicy = policy * legalMovesEncoded;
    return filteredPolicy / filteredPolicy.sum();
}

std::vector<MoveScore> mapPolicyToMoves(const torch::Tensor &policy, const Board *board) {
    const torch::Tensor nonzeroIndices = torch::nonzero(policy > 0);
    const size_t n = nonzeroIndices.size(0);

    std::vector<int> encodedMoves;
    encodedMoves.reserve(n);
    for (const size_t i : range(n)) {
        const int encodedMove = nonzeroIndices[i].item<int>();
        encodedMoves.push_back(encodedMove);
    }

    std::vector<Move> decodedMoves = decodeMoves(encodedMoves, board);

    std::vector<MoveScore> movesWithProbabilities;
    movesWithProbabilities.reserve(n);

    for (const size_t i : range(n)) {
        const float probability = policy[nonzeroIndices[i].item<int>()].item<float>();
        movesWithProbabilities.emplace_back(decodedMoves[i], probability);
    }

    return movesWithProbabilities;
}

std::vector<MoveScore> filterPolicyThenGetMovesAndProbabilities(const torch::Tensor &policy,
                                                                const Board *board) {
    // Gets a list of moves with their corresponding probabilities from a policy.
    //
    // The policy is a 1D tensor representing the probabilities of each move
    // in the board. The list of moves is a list of tuples, where each tuple contains
    // a move and its corresponding probability.
    //
    // param policy: The policy to get the moves and probabilities from.
    // param board: The chess board to filter the policy with.
    // :return: The list of moves with their corresponding probabilities.

    const torch::Tensor filteredPolicy = filterPolicyWithLegalMoves(policy, board);
    return mapPolicyToMoves(filteredPolicy, board);
}
