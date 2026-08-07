#include "games/chess/ChessBindings.hpp"

#include "games/chess/ChessEncoding.hpp"
#include "games/chess/ChessGameContract.hpp"

#include "bitboard.h"
#include "position.h"

#include <cstdint>
#include <string>

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_chess_search(py::module_ &module);
void bind_chess_analysis(py::module_ &module);

void bind_chess_game(py::module_ &module) {
    Stockfish::Bitboards::init();
    Stockfish::Position::init();

    module.def(
        "encode_board_packed_bytes",
        [](const std::string &fen) {
            const Board board(fen);
            const CompressedEncodedBoard encoded = ChessGameContract::encodeInput(board);
            std::string payload(CompressedEncodedBoard::packed_bytes, '\0');
            writePackedPlaneEncoding(encoded, reinterpret_cast<std::int8_t *>(payload.data()));
            return py::bytes(payload);
        },
        py::arg("fen"),
        R"pbdoc(Encode a FEN into the canonical packed plane-major byte layout.)pbdoc");

    bind_chess_search(module);
    bind_chess_analysis(module);
}
