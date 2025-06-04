#include "common.hpp"

#include "main.hpp"

#include "BoardEncoding.hpp"

#include <pybind11/pybind11.h>

#include <pybind11/numpy.h> // for numpy array support
#include <pybind11/stl.h>   // for automatic binding of STL containers

namespace py = pybind11;
using namespace py::literals;

auto encode_board(const std::string &fen) {
    Board board = Board::fromFEN(fen);
    if (!board.isValid()) {
        throw std::runtime_error("Invalid Board fen: " + fen);
    }
    return encodeBoard(board);
}

#include "bitboard.h"
#include "misc.h"
#include "position.h"
#include "tune.h"
#include "types.h"
#include "uci.h"

auto test() {

    Stockfish::Bitboards::init();
    Stockfish::Position::init();

    Stockfish::Position pos;
    Stockfish::StateInfo si;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, &si);

    // print all legal moves
    Stockfish::MoveList<GenType::LEGAL> moves(pos);
    for (const auto &move : moves) {
        std::cout << toString(move) << std::endl;
    }
}

PYBIND11_MODULE(AlphaZeroCpp, m) {
    m.doc() = "AlphaZero C++ bindings for Python.";

    m.def("test", &test, "Test function to initialize Bitboards and Position");

    m.def("self_play_main", &selfPlayMain,
          "Runs self-play on the given model save directory path and parameters", py::arg("run_id"),
          py::arg("save_path"), py::arg("num_processors"), py::arg("num_gpus"))
        .attr("__annotations__") =
        py::dict("return"_a = "None", "run_id"_a = "int", "save_path"_a = "str",
                 "num_processors"_a = "int", "num_gpus"_a = "int");

    m.def("board_inference_main", &boardInferenceMain,
          "Runs board inference on a list of FEN strings", py::arg("model_path"), py::arg("fens"))
        .attr("__annotations__") =
        py::dict("return"_a = "List[Tuple[float, List[Tuple[int, int]]]]", "model_path"_a = "str",
                 "fens"_a = "List[str]");

    // type alias for CompressedEncodedBoard = List[uint64]
    // type alias for EncodedBoard = List[List[List[int8]]]

    // encodeBoard function
    m.def("encode_board", &encode_board, "Encodes a board into a CompressedEncodedBoard",
          py::arg("fen"))
        .attr("__annotations__") = py::dict("return"_a = "CompressedEncodedBoard", "fen"_a = "str");
}
