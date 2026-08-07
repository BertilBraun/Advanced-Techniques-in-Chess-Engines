#include "common.hpp"

#include "games/chess/ChessGame.hpp"
#include "games/chess/presentation/ChessSearchPresentation.hpp"
#include "search/SelfPlay.hpp"
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
using Request = SelfPlaySearchRequest<ChessGame>;
using Result = SelfPlaySearchResult<ChessGame>;
using Batch = SelfPlaySearchBatch<ChessGame>;
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

    py::class_<ChessSearchRoot>(module, "ChessSearchRoot")
        .def_property_readonly("fen",
                               [](const ChessSearchRoot &root) { return root.position().fen(); })
        .def_property_readonly("visits", &ChessSearchRoot::visits)
        .def_property_readonly(
            "virtual_loss",
            [](const ChessSearchRoot &root) { return root.tree().root().virtual_loss; })
        .def_property_readonly(
            "result_sum", [](const ChessSearchRoot &root) { return root.tree().root().value_sum; })
        .def_property_readonly("is_terminal", &ChessSearchRoot::isTerminal)
        .def_property_readonly(
            "repetition_count",
            [](const ChessSearchRoot &root) { return root.position().repetitionCount(); })
        .def_property_readonly(
            "is_expanded",
            [](const ChessSearchRoot &root) { return root.tree().root().expanded(); })
        .def_property_readonly(
            "max_depth", [](const ChessSearchRoot &root) { return root.tree().maximumDepth(); })
        .def_property_readonly("children", &chessSearchChildren)
        .def_property_readonly("live_nodes", &ChessSearchRoot::liveNodeCount)
        .def_property_readonly(
            "total_child_records",
            [](const ChessSearchRoot &root) { return root.tree().totalChildCount(); })
        .def_property_readonly("arena_capacity",
                               [](const ChessSearchRoot &root) { return root.tree().capacity(); })
        .def("make_new_root", &rerootChessSearch, py::arg("child_index"))
        .def("reset", &ChessSearchRoot::reset)
        .def(
            "discount",
            [](ChessSearchRoot &root, const float retainedFraction) {
                root.tree().discount(retainedFraction);
            },
            py::arg("percentage_of_node_visits_to_keep"))
        .def("__repr__", &describeChessSearchRoot);

    py::class_<Request>(module, "ChessSelfPlaySearchRequest")
        .def(py::init<ChessSearchRoot, bool>(), py::arg("root"), py::arg("full_search"))
        .def_readonly("root", &Request::root)
        .def_readonly("full_search", &Request::full_search);

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

    py::class_<Result>(module, "ChessSelfPlaySearchResult")
        .def_readonly("root_value", &Result::root_value)
        .def_property_readonly("visits",
                               [](const Result &result) {
                                   std::vector<std::pair<int, int>> visits;
                                   visits.reserve(result.visits.size());
                                   for (const GameSearchVisit &visit : result.visits) {
                                       visits.emplace_back(visit.action_id,
                                                           static_cast<int>(visit.visit_count));
                                   }
                                   return visits;
                               })
        .def_readonly("root", &Result::root);

    py::class_<Batch>(module, "ChessSelfPlaySearchBatch")
        .def_readonly("results", &Batch::results)
        .def_readonly("statistics", &Batch::statistics)
        .def_readonly("simulations_completed", &Batch::simulations_completed);

    py::class_<Search>(module, "ChessSelfPlaySearch")
        .def(py::init<const InferenceConfiguration &, const SelfPlaySearchParameters &,
                      BatchedInferenceParameters, std::uint64_t>(),
             py::arg("runtime_parameters"), py::arg("search_parameters"),
             py::arg("inference_parameters"), py::arg("initial_model_version") = 0)
        .def_property_readonly("arena_capacity", &Search::arenaCapacity)
        .def_property_readonly("model_version", &Search::modelGeneration)
        .def(
            "new_root",
            [](const Search &search, const std::string &fen) { return search.newRoot(Board(fen)); },
            py::arg("fen"))
        .def(
            "new_root_with_history",
            [](const Search &search, const std::string &startingFen,
               const std::vector<std::string> &movesUci) {
                return search.newRoot(Board::replay(startingFen, movesUci));
            },
            py::arg("starting_fen"), py::arg("moves_uci"))
        .def("inference_statistics", &Search::inferenceStatistics)
        .def("refresh_model", &Search::refreshModel, py::arg("model_version"),
             py::arg("model_path"), py::call_guard<py::gil_scoped_release>())
        .def("update_search_schedule", &Search::updateSearchSchedule, py::arg("search_parameters"))
        .def("search", &Search::search, py::arg("requests"), py::arg("collect_statistics") = false)
        .def("inference_with_history", &inferenceWithHistory, py::arg("histories"),
             py::call_guard<py::gil_scoped_release>());
}
