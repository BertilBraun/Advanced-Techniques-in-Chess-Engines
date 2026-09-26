#include "TestRunner.hpp"
#include "games/chess/encoding/ChessEncoding.hpp"

namespace {
// Lc0 INPUT_CLASSICAL_112_PLANE: history position h (0 = current) owns planes 13h..13h+12, ours
// P N B R Q K, theirs P N B R Q K, then repetition.
constexpr int historyBase(const int history) { return history * 13; }
constexpr int ownPawns = 0;
constexpr int opponentPawns = 6;
constexpr int repetition = 12;
constexpr int ownQueenside = 104;
constexpr int ownKingside = 105;
constexpr int opponentQueenside = 106;
constexpr int opponentKingside = 107;
constexpr int blackToMove = 108;
constexpr int rule50 = 0;
constexpr int zeros = 1;
constexpr int ones = 2;
constexpr std::uint64_t rankTwo = 0x0000'0000'0000'FF00ULL;
constexpr std::uint64_t rankSeven = 0x00FF'0000'0000'0000ULL;

Board afterMoves(std::initializer_list<const char *> moves) {
    Board board;
    for (const char *move : moves) {
        board.makeMove(board.legalMoveFromUci(move));
    }
    return board;
}
} // namespace

int runChessEncodingTests() {
    Stockfish::Bitboards::init();
    Stockfish::Position::init();

    const Board midgame("r3k2r/ppp2ppp/2n1bn2/3qp3/3P4/2N1BN2/PPP2PPP/R2Q1RK1 w kq - 7 11");
    const torch::Tensor expected = tensorEncoding(encodeBoard(midgame));
    std::vector<std::int8_t> actual(static_cast<size_t>(expected.numel()));
    ChessEncoding::encodeInputInto(midgame, actual.data());
    const torch::Tensor actualTensor = torch::from_blob(actual.data(), expected.sizes(),
                                                        torch::TensorOptions().dtype(torch::kInt8));
    if (!torch::equal(expected, actualTensor)) {
        return 1;
    }

    Board historyBoard;
    historyBoard.makeMove(historyBoard.legalMoveFromUci("e2e4"));
    const ProjectEncodedBoard blackHistory = encodeProjectBoard(historyBoard);
    if (!blackHistory.binaryPlanes[22].test(52) || !blackHistory.binaryPlanes[23].test(36)) {
        return 1;
    }
    historyBoard.makeMove(historyBoard.legalMoveFromUci("c7c5"));
    const ProjectEncodedBoard whiteHistory = encodeProjectBoard(historyBoard);
    if (!whiteHistory.binaryPlanes[22].test(50) || !whiteHistory.binaryPlanes[23].test(34) ||
        !whiteHistory.binaryPlanes[24].test(12) || !whiteHistory.binaryPlanes[25].test(28)) {
        return 1;
    }
    if (whiteHistory.binaryPlanes[38].word(0) != 0xAA55'AA55'AA55'AA55ULL) {
        return 1;
    }
    const Board oppositeBishops("2b3k1/8/8/8/8/8/8/2B3K1 w - - 0 1");
    if (encodeProjectBoard(oppositeBishops).binaryPlanes[39].count() != 64) {
        return 1;
    }
    const ProjectEncodedBoard projectInitial = encodeProjectBoard(Board{});
    if (projectInitial.scalarPlanes[7] != 8 || projectInitial.scalarPlanes[8] != 2 ||
        projectInitial.scalarPlanes[9] != 2 || projectInitial.scalarPlanes[10] != 2 ||
        projectInitial.scalarPlanes[11] != 1) {
        return 1;
    }

    const Lc0EncodedBoard initial = encodeLc0Board(Board{});
    if (initial.binaryPlanes[ownPawns].word(0) != rankTwo ||
        initial.binaryPlanes[opponentPawns].word(0) != rankSeven) {
        return 1;
    }
    for (int history = 1; history < 8; ++history) {
        for (int plane = 0; plane < 13; ++plane) {
            if (!initial.binaryPlanes[historyBase(history) + plane].none()) {
                return 1;
            }
        }
    }
    if (initial.binaryPlanes[ownQueenside].count() != 64 ||
        initial.binaryPlanes[ownKingside].count() != 64 ||
        initial.binaryPlanes[opponentQueenside].count() != 64 ||
        initial.binaryPlanes[opponentKingside].count() != 64 ||
        !initial.binaryPlanes[blackToMove].none()) {
        return 1;
    }
    if (initial.scalarPlanes[rule50] != 0 || initial.scalarPlanes[zeros] != 0 ||
        initial.scalarPlanes[ones] != 1) {
        return 1;
    }

    // Black to move after e2e4: the board is mirrored and colours swapped, so White's advanced pawn
    // appears on e5 (square 36) and the vacated e2 as e7 (square 52).
    const Lc0EncodedBoard afterE4 = encodeLc0Board(afterMoves({"e2e4"}));
    if (afterE4.binaryPlanes[ownPawns].word(0) != rankTwo ||
        !afterE4.binaryPlanes[opponentPawns].test(36) ||
        afterE4.binaryPlanes[opponentPawns].test(52) ||
        afterE4.binaryPlanes[blackToMove].count() != 64) {
        return 1;
    }
    // The previous position, seen from the same side, still has White's e-pawn at home.
    if (!afterE4.binaryPlanes[historyBase(1) + opponentPawns].test(52) ||
        afterE4.binaryPlanes[historyBase(1) + opponentPawns].test(36) ||
        afterE4.binaryPlanes[historyBase(1) + ownPawns].word(0) != rankTwo) {
        return 1;
    }
    if (!afterE4.binaryPlanes[historyBase(2) + ownPawns].none()) {
        return 1;
    }

    // Unlike the repetition chain, the history window survives a pawn move and a capture.
    const Lc0EncodedBoard afterCapture =
        encodeLc0Board(afterMoves({"e2e4", "d7d5", "e4d5", "d8d5"}));
    for (int history = 0; history < 5; ++history) {
        if (afterCapture.binaryPlanes[historyBase(history) + ownPawns].none()) {
            return 1;
        }
    }
    if (!afterCapture.binaryPlanes[historyBase(5) + ownPawns].none()) {
        return 1;
    }

    const Lc0EncodedBoard knightShuffle =
        encodeLc0Board(afterMoves({"g1f3", "g8f6", "f3g1", "f6g8"}));
    if (knightShuffle.binaryPlanes[repetition].count() != 64 ||
        !knightShuffle.binaryPlanes[historyBase(1) + repetition].none()) {
        return 1;
    }

    const Lc0EncodedBoard asymmetricCastling =
        encodeLc0Board(Board("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 17 30"));
    if (!asymmetricCastling.binaryPlanes[ownQueenside].none() ||
        asymmetricCastling.binaryPlanes[ownKingside].count() != 64 ||
        asymmetricCastling.binaryPlanes[opponentQueenside].count() != 64 ||
        !asymmetricCastling.binaryPlanes[opponentKingside].none() ||
        asymmetricCastling.scalarPlanes[rule50] != 17) {
        return 1;
    }
    return 0;
}
