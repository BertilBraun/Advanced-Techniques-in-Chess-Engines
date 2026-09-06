#include "TestRunner.hpp"
#include "games/chess/encoding/ChessEncoding.hpp"

int runChessEncodingTests() {
    Stockfish::Bitboards::init();
    Stockfish::Position::init();
    const Board board("r3k2r/ppp2ppp/2n1bn2/3qp3/3P4/2N1BN2/PPP2PPP/R2Q1RK1 w kq - 7 11");
    const CompressedEncodedBoard compressed = encodeBoard(board);
    const torch::Tensor expected = tensorEncoding(compressed);
    std::vector<std::int8_t> actual(static_cast<size_t>(expected.numel()));
    ChessEncoding::encodeInputInto(board, actual.data());
    const torch::Tensor actualTensor = torch::from_blob(actual.data(), expected.sizes(),
                                                        torch::TensorOptions().dtype(torch::kInt8));
    if (!torch::equal(expected, actualTensor)) {
        return 1;
    }
    Board historyBoard;
    historyBoard.makeMove(historyBoard.legalMoveFromUci("e2e4"));
    const CompressedEncodedBoard blackHistory = encodeBoard(historyBoard);
    if (!blackHistory.binaryPlanes[22].test(52) || !blackHistory.binaryPlanes[23].test(36)) {
        return 1;
    }
    historyBoard.makeMove(historyBoard.legalMoveFromUci("c7c5"));
    const CompressedEncodedBoard whiteHistory = encodeBoard(historyBoard);
    if (!whiteHistory.binaryPlanes[22].test(50) || !whiteHistory.binaryPlanes[23].test(34) ||
        !whiteHistory.binaryPlanes[24].test(12) || !whiteHistory.binaryPlanes[25].test(28)) {
        return 1;
    }
    if (whiteHistory.binaryPlanes[38].word(0) != 0xAA55'AA55'AA55'AA55ULL) {
        return 1;
    }
    const Board oppositeBishops("2b3k1/8/8/8/8/8/8/2B3K1 w - - 0 1");
    if (encodeBoard(oppositeBishops).binaryPlanes[39].count() != 64) {
        return 1;
    }
    const CompressedEncodedBoard initial = encodeBoard(Board{});
    if (initial.scalarPlanes[7] != 8 || initial.scalarPlanes[8] != 2 ||
        initial.scalarPlanes[9] != 2 || initial.scalarPlanes[10] != 2 ||
        initial.scalarPlanes[11] != 1) {
        return 1;
    }
    return 0;
}
