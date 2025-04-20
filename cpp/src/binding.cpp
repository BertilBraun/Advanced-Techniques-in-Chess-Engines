#include "common.hpp"

#include "main.hpp"

#include "BoardEncoding.hpp"
#include "MCTS/VisitCounts.hpp"
#include "MoveEncoding.hpp"
#include "eval/ModelEvaluation.hpp"

#include <pybind11/pybind11.h>

#include <pybind11/numpy.h> // for numpy array support
#include <pybind11/stl.h>   // for automatic binding of STL containers

namespace py = pybind11;
using namespace py::literals;

py::object cpp_move_to_py(const Move &move) {
    // Convert the C++ Move to a UCI string.
    std::string uci = move.uci();

    // Import the python-chess module.
    py::module_ chess_mod = py::module_::import("chess");
    // Call chess.Move.from_uci(uci) to create a python-chess Move.
    py::object py_move = chess_mod.attr("Move").attr("from_uci")(uci);
    return py_move;
}

Move py_move_to_cpp(const py::object &move) {
    // Convert the python-chess Move to a UCI string.
    std::string uci = move.attr("uci")().cast<std::string>();
    // Convert the UCI string to a C++ move.
    Move cpp_move = Move::fromUci(uci);
    return cpp_move;
}

py::object eval_board_iterate(const std::string &modelPath, const std::string &fen,
                              bool networkOnly, float maxTime) {
    Move bestMove = evalBoardIterate(modelPath, fen, networkOnly, maxTime);
    return cpp_move_to_py(bestMove);
}

int encode_move(const py::object &move) {
    Move cpp_move = py_move_to_cpp(move);
    return encodeMove(cpp_move);
}

py::object decode_move(int encoded) {
    Move cpp_move = decodeMove(encoded);
    return cpp_move_to_py(cpp_move);
}

auto encode_board(const std::string &fen) {
    Board board = Board::fromFEN(fen);
    if (!board.isValid()) {
        throw std::runtime_error("Invalid Board fen: " + fen);
    }
    return encodeBoard(board);
}

auto evaluate_model(int iteration, std::string const &save_path, int num_games, int runId,
                    std::string const &model_path) {
    TensorBoardLogger logger = getTensorBoardLogger(runId);
    ModelEvaluation evaluator(iteration, save_path, num_games, &logger);
    Results res = evaluator.play_two_models_search(model_path);
    return py::make_tuple(res.wins, res.losses, res.draws);
}

PYBIND11_MODULE(AlphaZeroCpp, m) {
    m.doc() = "AlphaZero C++ bindings for Python.";

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

    m.def("eval_board_iterate", &eval_board_iterate, "Runs board inference on a single FEN string",
          py::arg("model_path"), py::arg("fen"), py::arg("network_only") = false,
          py::arg("max_time") = 5.0)
        .attr("__annotations__") =
        py::dict("return"_a = "chess.Move", "model_path"_a = "str", "fen"_a = "str",
                 "network_only"_a = "bool", "max_time"_a = "float");

    m.def("evaluate_model", &evaluate_model, "Evaluates a model against the current model",
          py::arg("iteration"), py::arg("save_path"), py::arg("num_games"), py::arg("run_id"),
          py::arg("model_path"))
        .attr("__annotations__") =
        py::dict("return"_a = "Tuple[int, int, int]", "iteration"_a = "int", "save_path"_a = "str",
                 "num_games"_a = "int", "run_id"_a = "int", "model_path"_a = "str");

    // type alias for CompressedEncodedBoard = List[uint64]
    // type alias for EncodedBoard = List[List[List[int8]]]

    // encodeBoard function
    m.def("encode_board", &encode_board, "Encodes a board into a CompressedEncodedBoard",
          py::arg("fen"))
        .attr("__annotations__") = py::dict("return"_a = "CompressedEncodedBoard", "fen"_a = "str");

    // decompress function
    m.def("decompress", &decompress, "Decompresses a CompressedEncodedBoard into an EncodedBoard",
          py::arg("compressed"))
        .attr("__annotations__") =
        py::dict("return"_a = "EncodedBoard", "compressed"_a = "CompressedEncodedBoard");

    // encodeMove function
    m.def("encode_move", &encode_move, "Encodes a move into an integer", py::arg("move"))
        .attr("__annotations__") = py::dict("return"_a = "int", "move"_a = "chess.Move");

    // decodeMove function
    m.def("decode_move", &decode_move, "Decodes an integer into a move", py::arg("move_index"))
        .attr("__annotations__") = py::dict("return"_a = "chess.Move", "move_index"_a = "int");

    // actionProbabilities function
    m.def("action_probabilities", &actionProbabilities<int>,
          "Returns the action probabilities for a given visit counts vector",
          py::arg("visit_counts"))
        .attr("__annotations__") =
        py::dict("return"_a = "List[int]", "visit_counts"_a = "List[Tuple[int, int]]");
}
