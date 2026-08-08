#include "common.hpp"

#include "games/chess/ChessGame.hpp"
#include "games/chess/presentation/ChessSearchPresentation.hpp"
#include "search/SelfPlayBindings.hpp"
#include "util/py.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {
using Search = GameSelfPlaySearch<ChessGame>;
using InferenceResult = SearchInferenceResult<ChessGame>;
using EncodedInference = std::pair<std::vector<std::pair<int, float>>, float>;

[[nodiscard]] EncodedInference encodeInference(const Board &board,
                                               const InferenceResult &inferenceResult) {
    std::vector<std::pair<int, float>> encodedMoves;
    encodedMoves.reserve(inferenceResult.actions.size());
    for (const auto &[action, score] : inferenceResult.actions) {
        encodedMoves.emplace_back(ChessGame::Encoding::actionId(action, board), score);
    }
    return {encodedMoves, inferenceResult.value()};
}

[[nodiscard]] std::vector<EncodedInference> inferenceWithHistory(
    Search &search,
    const std::vector<std::pair<std::string, std::vector<std::string>>> &histories) {
    std::vector<Board> boards;
    boards.reserve(histories.size());
    for (const auto &[startingFen, movesUci] : histories) {
        boards.push_back(Board::replay(startingFen, movesUci));
    }
    const std::vector<InferenceResult> inferenceResults = search.evaluate(boards);
    std::vector<EncodedInference> encodedResults;
    encodedResults.reserve(boards.size());
    for (const auto index : range(boards.size())) {
        encodedResults.push_back(encodeInference(boards[index], inferenceResults[index]));
    }
    return encodedResults;
}
} // namespace

void bind_chess_search(py::module_ &module) {
    py::class_<ChessSearchChild>(module, "ChessSearchChild")
        .def_readonly("move", &ChessSearchChild::move)
        .def_readonly("encoded_move", &ChessSearchChild::encoded_move)
        .def_readonly("raw_policy", &ChessSearchChild::raw_policy)
        .def_readonly("policy", &ChessSearchChild::policy)
        .def_readonly("visits", &ChessSearchChild::visits)
        .def_readonly("result_sum", &ChessSearchChild::result_sum)
        .def_readonly("virtual_loss", &ChessSearchChild::virtual_loss)
        .def_readonly("is_materialized", &ChessSearchChild::is_materialized);

    auto bindings = bindSelfPlay<ChessGame>(
        module, {.root = "ChessSearchRoot",
                 .search = "ChessSelfPlaySearch",
                 .request = "ChessSelfPlaySearchRequest",
                 .result = "ChessSelfPlaySearchResult",
                 .batch = "ChessSelfPlaySearchBatch"});
    bindings.root
        .def_property_readonly("fen",
                               [](const ChessSearchRoot &root) { return root.position().fen(); })
        .def_property_readonly(
            "virtual_loss",
            [](const ChessSearchRoot &root) { return root.tree().root().virtual_loss; })
        .def_property_readonly(
            "result_sum", [](const ChessSearchRoot &root) { return root.tree().root().value_sum; })
        .def_property_readonly(
            "repetition_count",
            [](const ChessSearchRoot &root) { return root.position().repetitionCount(); })
        .def_property_readonly(
            "is_expanded",
            [](const ChessSearchRoot &root) { return root.tree().root().expanded(); })
        .def_property_readonly(
            "max_depth", [](const ChessSearchRoot &root) { return root.tree().maximumDepth(); })
        .def_property_readonly("children", &chessSearchChildren)
        .def_property_readonly(
            "total_child_records",
            [](const ChessSearchRoot &root) { return root.tree().totalChildCount(); })
        .def_property_readonly("arena_capacity",
                               [](const ChessSearchRoot &root) { return root.tree().capacity(); })
        .def("make_new_root", &rerootChessSearch, py::arg("child_index"))
        .def("__repr__", &describeChessSearchRoot);

    module.def(
        "new_root",
        [](const std::string &fen, const std::uint32_t arenaCapacity) {
            return createChessSearchRoot(Board(fen), arenaCapacity);
        },
        py::arg("fen"), py::arg("arena_capacity"));
    module.def(
        "new_root_with_history",
        [](const std::string &startingFen, const std::vector<std::string> &movesUci,
           const std::uint32_t arenaCapacity) {
            return createChessSearchRoot(Board::replay(startingFen, movesUci), arenaCapacity);
        },
        py::arg("starting_fen"), py::arg("moves_uci"), py::arg("arena_capacity"));

    bindings.search
        .def_property_readonly("arena_capacity", &Search::arenaCapacity)
        .def_property_readonly("model_version", &Search::modelGeneration)
        .def(
            "new_root_with_history",
            [](const Search &search, const std::string &startingFen,
               const std::vector<std::string> &movesUci) {
                return search.newRoot(Board::replay(startingFen, movesUci));
            },
            py::arg("starting_fen"), py::arg("moves_uci"))
        .def("inference_with_history", &inferenceWithHistory, py::arg("histories"),
             py::call_guard<py::gil_scoped_release>());
}
