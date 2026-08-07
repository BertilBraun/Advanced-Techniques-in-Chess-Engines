#include "games/chess/ChessBindings.hpp"

#include "common.hpp"

#include "InteractiveEngine.hpp"
#include "games/chess/ChessSelfPlaySearch.hpp"
#include "games/chess/ChessAction.hpp"
#include "games/chess/ChessEncoding.hpp"
#include "games/chess/ChessGameContract.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using EncodedInference = std::pair<std::vector<std::pair<int, float>>, float>;

EncodedInference encodeInference(const Board &board, const ChessInferenceResult &inferenceResult) {
    std::vector<std::pair<int, float>> encodedMoves;
    encodedMoves.reserve(inferenceResult.actions.size());
    for (const auto &[move, score] : inferenceResult.actions) {
        encodedMoves.emplace_back(encodeMove(move, &board), score);
    }
    return {encodedMoves, inferenceResult.value()};
}

std::vector<EncodedInference> inferenceWithHistory(
    ChessSelfPlaySearch &search,
    const std::vector<std::pair<std::string, std::vector<std::string>>> &histories) {
    std::vector<Board> boards;
    boards.reserve(histories.size());
    for (const auto &[startingFen, movesUci] : histories) {
        boards.push_back(ChessGameContract::replayPosition(startingFen, movesUci));
    }
    std::vector<const Board *> boardPointers;
    boardPointers.reserve(boards.size());
    for (const Board &board : boards) {
        boardPointers.push_back(&board);
    }
    const std::vector<ChessInferenceResult> inferenceResults = search.evaluate(boardPointers);
    std::vector<EncodedInference> encodedResults;
    encodedResults.reserve(boards.size());
    for (std::size_t index = 0; index < boards.size(); ++index) {
        encodedResults.push_back(encodeInference(boards[index], inferenceResults[index]));
    }
    return encodedResults;
}

// ——————————————————————————————————————————————
// Bind everything with pybind11:
void bind_chess_game(py::module_ &m) {

    py::class_<ChessSearchChild>(m, "ChessSearchChild")
        .def_readonly("move", &ChessSearchChild::move)
        .def_readonly("encoded_move", &ChessSearchChild::encoded_move)
        .def_readonly("raw_policy", &ChessSearchChild::raw_policy)
        .def_readonly("policy", &ChessSearchChild::policy)
        .def_readonly("visits", &ChessSearchChild::visits)
        .def_readonly("result_sum", &ChessSearchChild::result_sum)
        .def_readonly("virtual_loss", &ChessSearchChild::virtual_loss)
        .def_readonly("is_materialized", &ChessSearchChild::is_materialized);

    py::class_<ChessSearchRoot>(m, "ChessSearchRoot")
        .def_property_readonly("fen", [](const ChessSearchRoot &root) { return root.board().fen(); })
        .def_property_readonly("move", &ChessSearchRoot::move)
        .def_property_readonly("visits", &ChessSearchRoot::visits)
        .def_property_readonly("virtual_loss", &ChessSearchRoot::virtualLoss)
        .def_property_readonly("result_sum", &ChessSearchRoot::resultSum)
        .def_property_readonly("is_terminal", &ChessSearchRoot::isTerminal)
        .def_property_readonly("repetition_count",
                               [](const ChessSearchRoot &root) { return root.board().repetitionCount(); })
        .def_property_readonly("is_expanded", &ChessSearchRoot::isExpanded)
        .def_property_readonly("max_depth", &ChessSearchRoot::maxDepth)
        .def_property_readonly("children", &ChessSearchRoot::children)
        .def_property_readonly("live_nodes", &ChessSearchRoot::liveNodeCount)
        .def_property_readonly("total_child_records", &ChessSearchRoot::totalChildCount)
        .def_property_readonly("arena_capacity", &ChessSearchRoot::arenaCapacity)
        .def("make_new_root", &ChessSearchRoot::makeNewRoot, py::arg("child_index"),
             R"pbdoc(
            Prune the old tree and return a new root node.
            `child_index` is the index of the child to make the new root.
            )pbdoc")
        .def(
            "reset", &ChessSearchRoot::reset,
            R"pbdoc(Discard logical search state while retaining reusable arena allocations.)pbdoc")
        .def("discount", &ChessSearchRoot::discount, py::arg("percentage_of_node_visits_to_keep"),
             R"pbdoc(
            Discount the node's score and visits by a percentage.
            Descendant materializations are explicitly pruned when required by the fixed arena.
            )pbdoc")
        .def("__repr__", &ChessSearchRoot::repr);

    py::class_<ChessSelfPlaySearchRequest>(m, "ChessSelfPlaySearchRequest")
        .def(py::init<ChessSearchRoot, bool>(), py::arg("root"), py::arg("full_search"))
        .def_readonly("root", &ChessSelfPlaySearchRequest::root)
        .def_readonly("full_search", &ChessSelfPlaySearchRequest::full_search);

    m.def(
        "new_root",
        [](const std::string &fen, const uint32 arenaCapacity) {
            return ChessSearchRoot::create(fen, arenaCapacity);
        },
        py::arg("fen"), py::arg("arena_capacity"),
        R"pbdoc(
            Create a self-play MCTS root with an explicit fixed arena capacity.
          )pbdoc");
    m.def(
        "new_root_with_history",
        [](const std::string &startingFen, const std::vector<std::string> &movesUci,
           const uint32 arenaCapacity) {
            return ChessSearchRoot::create(
                ChessGameContract::replayPosition(startingFen, movesUci), arenaCapacity);
        },
        py::arg("starting_fen"), py::arg("moves_uci"), py::arg("arena_capacity"),
        R"pbdoc(Create a fixed-capacity MCTS root by replaying bounded UCI history.)pbdoc");

    m.def(
        "encode_board_packed_bytes",
        [](const std::string &fen) {
            const Board board(fen);
            const CompressedEncodedBoard encoded = ChessGameContract::encodeInput(board);
            std::string payload;
            payload.resize(CHESS_PACKED_BYTES);
            writePackedPlaneEncoding(encoded, reinterpret_cast<int8 *>(payload.data()));
            return py::bytes(payload);
        },
        py::arg("fen"),
        R"pbdoc(Encode a FEN into the canonical packed plane-major byte layout.)pbdoc");

    py::class_<ChessSelfPlaySearchParameters>(m, "ChessSelfPlaySearchParameters")
        .def(py::init<int, uint32, uint32, float, float, float, uint8>(),
             py::arg("parallel_searches"), py::arg("full_searches"),
             py::arg("fast_searches"), py::arg("exploration_constant"),
             py::arg("dirichlet_alpha"), py::arg("dirichlet_epsilon"),
             py::arg("minimum_root_visits"))
        .def_readwrite("parallel_searches", &ChessSelfPlaySearchParameters::parallel_searches)
        .def_readwrite("full_searches", &ChessSelfPlaySearchParameters::full_searches)
        .def_readwrite("fast_searches", &ChessSelfPlaySearchParameters::fast_searches)
        .def_readwrite("exploration_constant",
                       &ChessSelfPlaySearchParameters::exploration_constant)
        .def_readwrite("dirichlet_alpha", &ChessSelfPlaySearchParameters::dirichlet_alpha)
        .def_readwrite("dirichlet_epsilon", &ChessSelfPlaySearchParameters::dirichlet_epsilon)
        .def_readwrite("minimum_root_visits",
                       &ChessSelfPlaySearchParameters::minimum_root_visits);

    py::class_<ChessSelfPlaySearchResult>(m, "ChessSelfPlaySearchResult")
        .def_readonly("root_value", &ChessSelfPlaySearchResult::root_value)
        .def_readonly("visits", &ChessSelfPlaySearchResult::visits)
        .def_readonly("root", &ChessSelfPlaySearchResult::root);

    py::class_<ChessSelfPlaySearchStatistics>(m, "ChessSelfPlaySearchStatistics")
        .def_readonly("average_depth", &ChessSelfPlaySearchStatistics::average_depth)
        .def_readonly("average_entropy", &ChessSelfPlaySearchStatistics::average_entropy)
        .def_readonly("average_kl_divergence",
                      &ChessSelfPlaySearchStatistics::average_kl_divergence)
        .def_readonly("average_policy_search_kl_divergence",
                      &ChessSelfPlaySearchStatistics::average_policy_search_kl_divergence)
        .def_readonly("top_action_disagreement",
                      &ChessSelfPlaySearchStatistics::top_action_disagreement)
        .def_readonly("selected_action_prior_rank",
                      &ChessSelfPlaySearchStatistics::selected_action_prior_rank);

    py::class_<ChessSelfPlaySearchBatch>(m, "ChessSelfPlaySearchBatch")
        .def_readonly("results", &ChessSelfPlaySearchBatch::results)
        .def_readonly("statistics", &ChessSelfPlaySearchBatch::statistics)
        .def_readonly("simulations_completed",
                      &ChessSelfPlaySearchBatch::simulations_completed);

    py::class_<ChessSelfPlaySearch>(m, "ChessSelfPlaySearch")
        .def(py::init<const InferenceRuntimeParameters &, const ChessSelfPlaySearchParameters &,
                      BatchedInferenceParameters, uint64>(),
             py::arg("runtime_parameters"), py::arg("search_parameters"),
             py::arg("inference_parameters"),
             py::arg("initial_model_version") = 0)
        .def_property_readonly("arena_capacity", &ChessSelfPlaySearch::arenaCapacity)
        .def_property_readonly("model_version", &ChessSelfPlaySearch::modelVersion)
        .def(
            "new_root",
            [](const ChessSelfPlaySearch &self, const std::string &fen) {
                return self.newRoot(fen);
            },
            py::arg("fen"))
        .def(
            "new_root_with_history",
            [](const ChessSelfPlaySearch &self, const std::string &startingFen,
               const std::vector<std::string> &movesUci) {
                return self.newRoot(ChessGameContract::replayPosition(startingFen, movesUci));
            },
            py::arg("starting_fen"), py::arg("moves_uci"))
        .def("inference_statistics", &ChessSelfPlaySearch::inferenceStatistics)
        .def("refresh_model", &ChessSelfPlaySearch::refreshModel, py::arg("model_version"),
             py::arg("model_path"),
             py::call_guard<py::gil_scoped_release>())
        .def("update_search_schedule", &ChessSelfPlaySearch::updateSearchSchedule,
             py::arg("search_parameters"))
        .def("search", &ChessSelfPlaySearch::search, py::arg("requests"),
             py::arg("collect_statistics") = false,
             R"pbdoc(
                 Run batched chess self-play search requests.
                 Returns a ChessSelfPlaySearchBatch whose results contain:
                     - root_value: float
                     - visits: List of (encoded_move: int, visit_count: int)
                     - root: the retained ChessSearchRoot
             )pbdoc")
        .def("inference_with_history", &inferenceWithHistory, py::arg("histories"),
             py::call_guard<py::gil_scoped_release>());

    py::enum_<AnalysisMode>(m, "AnalysisMode")
        .value("POLICY", AnalysisMode::Policy)
        .value("MCTS", AnalysisMode::Mcts);

    py::class_<WdlPrediction>(m, "WdlPrediction")
        .def_readonly("win", &WdlPrediction::win)
        .def_readonly("draw", &WdlPrediction::draw)
        .def_readonly("loss", &WdlPrediction::loss)
        .def_property_readonly("value", &WdlPrediction::expectedValue);
    m.attr("OutcomeProbabilities") = m.attr("WdlPrediction");

    py::class_<CandidateAnalysis>(m, "CandidateAnalysis")
        .def_readonly("move_uci", &CandidateAnalysis::move_uci)
        .def_readonly("policy_prior", &CandidateAnalysis::policy_prior)
        .def_readonly("visits", &CandidateAnalysis::visits)
        .def_readonly("visit_share", &CandidateAnalysis::visit_share)
        .def_readonly("mean_value", &CandidateAnalysis::mean_value);

    py::class_<AnalysisResult>(m, "AnalysisResult")
        .def_readonly("chosen_move_uci", &AnalysisResult::chosen_move_uci)
        .def_readonly("value", &AnalysisResult::value)
        .def_readonly("outcome", &AnalysisResult::outcome)
        .def_readonly("candidates", &AnalysisResult::candidates)
        .def_readonly("searches", &AnalysisResult::searches)
        .def_readonly("maximum_depth", &AnalysisResult::maximum_depth)
        .def_readonly("elapsed_milliseconds", &AnalysisResult::elapsed_milliseconds)
        .def_readonly("principal_variation", &AnalysisResult::principal_variation);

    py::class_<InteractiveSearchParams>(m, "InteractiveSearchParams")
        .def(py::init<float, int, int, int>(), py::arg("exploration_constant"),
             py::arg("inference_workers"), py::arg("inference_batch_size"),
             py::arg("outstanding_batches_per_worker") = 2)
        .def_readwrite("exploration_constant", &InteractiveSearchParams::exploration_constant)
        .def_readwrite("inference_workers", &InteractiveSearchParams::inference_workers)
        .def_readwrite("inference_batch_size", &InteractiveSearchParams::inference_batch_size)
        .def_readwrite("outstanding_batches_per_worker",
                       &InteractiveSearchParams::outstanding_batches_per_worker);

    py::class_<InteractiveEngine, std::shared_ptr<InteractiveEngine>>(m, "InteractiveEngine")
        .def(py::init<const InferenceRuntimeParameters &, const InteractiveSearchParams &>(),
             py::arg("runtime_parameters"), py::arg("search_parameters"))
        .def("new_game", &InteractiveEngine::newGame, py::arg("starting_fen"), py::arg("moves_uci"))
        .def("get_inference_statistics", &InteractiveEngine::inferenceStatistics);

    py::class_<InteractiveGame, std::shared_ptr<InteractiveGame>>(m, "InteractiveGame")
        .def("apply_move", &InteractiveGame::applyMove, py::arg("move_uci"))
        .def("analyze", &InteractiveGame::analyze, py::arg("mode"),
             py::arg("time_limit_seconds") = std::nullopt, py::arg("search_limit") = std::nullopt,
             py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("fen", &InteractiveGame::fen)
        .def_property_readonly("starting_fen", &InteractiveGame::startingFen)
        .def_property_readonly("moves_uci", &InteractiveGame::movesUci)
        .def_property_readonly("root_visits", &InteractiveGame::rootVisits);
}
