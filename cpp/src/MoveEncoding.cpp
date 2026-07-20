#include "MoveEncoding.hpp"

namespace defines {
const std::vector<std::pair<int, int>> DIRECTIONS = {{1, 0},  {1, 1},   {0, 1},  {-1, 1},
                                                     {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
const std::vector<std::pair<int, int>> KNIGHT_MOVES = {{2, 1},   {1, 2},   {-1, 2}, {-2, 1},
                                                       {-2, -1}, {-1, -2}, {1, -2}, {2, -1}};
constexpr std::array PROMOTION_PIECES = {
    PieceType::QUEEN,
    PieceType::ROOK,
    PieceType::BISHOP,
    PieceType::KNIGHT,
};

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
        moveMappings[fromSquare][toSquare][static_cast<int>(promotionType)] = index++;
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
    const PieceType promotionType =
        move.type_of() == PROMOTION ? move.promotion_type() : PieceType::NO_PIECE_TYPE;

    if (board->currentPlayer() == -1) {
        // Flip the move to match the board's perspective.
        fromSquare = flip_rank(fromSquare);
        toSquare = flip_rank(toSquare);
    }

    const int actionIndex = MOVE_MAPPINGS[fromSquare][toSquare][promotionType];
    assert(0 <= actionIndex && actionIndex < ACTION_SIZE &&
           "Encoded move index out of bounds in encodeMove");
    return actionIndex;
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

    const std::vector<Move> &legalMoves = board->validMoves();

    std::vector<Move> moves;
    moves.reserve(moveIndices.size());
    for (const int moveIndex : moveIndices) {
        assert(0 <= moveIndex && moveIndex < ACTION_SIZE &&
               "Move index out of bounds in decodeMoves");

        auto [from_square, to_square, promotion_type] = REVERSE_MOVE_MAPPINGS[moveIndex];

        if (board->currentPlayer() == -1) {
            from_square = flip_rank(from_square);
            to_square = flip_rank(to_square);
        }

        bool found = false;
        for (Move move : legalMoves) {
            const bool isCorrectPosition =
                move.from_sq() == from_square && move.to_sq() == to_square;
            const bool isPromotion = move.type_of() == PROMOTION;
            const bool isCorrectPromotionType =
                (isPromotion && move.promotion_type() == promotion_type) ||
                (!isPromotion && promotion_type == PieceType::NO_PIECE_TYPE);
            // Check if the move matches the from and to square and the promotion type.
            // If the move is a promotion, we check if the promotion type matches.
            // If the move is not a promotion, we check if the promotion type is NO_PIECE_TYPE.
            if (isCorrectPosition &&

                isCorrectPromotionType) {
                // If the move matches the from and to square and the promotion type, we can use it.
                moves.emplace_back(move);
                found = true;
                break; // No need to check further, we found the move.
            }
        }
        assert(found && "Move not found in legal moves in decodeMoves");
    }
    return moves;
}

std::vector<EncodedMoveScore> filterPolicyThenGetMovesAndProbabilities(const torch::Tensor &policy,
                                                                       const Board *board) {
    assert(policy.device().is_cpu() && "Policy must reside on the CPU");
    assert(policy.scalar_type() == torch::kFloat32 && "Policy must use float32");
    assert(policy.dim() == 1 && policy.numel() == ACTION_SIZE &&
           "Policy must contain one score per encoded move");
    assert(policy.is_contiguous() && "Policy must be contiguous");

    const std::vector<Move> &legalMoves = board->validMoves();
    if (legalMoves.empty()) {
        return {};
    }

    const float *policyData = policy.data_ptr<float>();
    std::vector<EncodedMoveScore> movesWithProbabilities;
    movesWithProbabilities.reserve(legalMoves.size());

    for (const Move move : legalMoves) {
        const int encodedMove = encodeMove(move, board);
        const float score = policyData[encodedMove];
        movesWithProbabilities.emplace_back(encodedMove, score);
    }

    // Preserve the encoded-index order produced by the previous Torch implementation.
    std::sort(movesWithProbabilities.begin(), movesWithProbabilities.end(),
              [](const EncodedMoveScore &left, const EncodedMoveScore &right) {
                  return left.first < right.first;
              });

    float legalPolicySum = 0.0f;
    for (const EncodedMoveScore &moveWithProbability : movesWithProbabilities) {
        legalPolicySum += moveWithProbability.second;
    }

    if (legalPolicySum < 0.00001f) {
        const float uniformProbability = 1.0f / static_cast<float>(movesWithProbabilities.size());
        for (EncodedMoveScore &moveWithProbability : movesWithProbabilities) {
            moveWithProbability.second = uniformProbability;
        }
        return movesWithProbabilities;
    }

    for (EncodedMoveScore &moveWithProbability : movesWithProbabilities) {
        moveWithProbability.second /= legalPolicySum;
    }
    std::erase_if(movesWithProbabilities, [](const EncodedMoveScore &moveWithProbability) {
        return !(moveWithProbability.second > 0.0f);
    });
    return movesWithProbabilities;
}
