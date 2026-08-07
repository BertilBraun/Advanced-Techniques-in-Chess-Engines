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
    return 0;
}
