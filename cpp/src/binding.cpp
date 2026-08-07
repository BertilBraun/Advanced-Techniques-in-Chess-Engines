#include "common.hpp"

#include "InteractiveEngine.hpp"
#include "MCTS/MCTS.hpp"
#include "games/chess/ChessAction.hpp"
#include "games/chess/ChessEncoding.hpp"
#include "games/chess/ChessGameContract.hpp"
#include "games/go/GoBindings.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

static void init() {
    // We need to initialize the Stockfish engine before using it.
    Bitboards::init();
    Position::init();

    torch::set_num_threads(1); // Set the number of threads for PyTorch to 1 to avoid conflicts.
    torch::set_num_interop_threads(1); // inter-op
    setenv("OMP_NUM_THREADS", "1", 1); // for MKL / OpenBLAS just in case
    setenv("MKL_NUM_THREADS", "1", 1);
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
}

std::pair<std::vector<std::pair<int, float>>, float> inference(MCTS &self, const std::string &fen) {
    const Board board(fen);

    std::vector<const Board *> boards;
    boards.push_back(&board);

    const auto result = self.inferenceBatch(boards);
    assert(result.size() == 1 && "Inference should return exactly one result for one board");
    const InferenceResult &inferenceResult = result[0];

    std::vector<std::pair<int, float>> encodedMoves;
    encodedMoves.reserve(inferenceResult.actions.size());
    for (const auto &[move, score] : inferenceResult.actions) {
        encodedMoves.emplace_back(encodeMove(move, &board), score);
    }

    return {encodedMoves, inferenceResult.value()};
}

// ——————————————————————————————————————————————
// Bind everything with pybind11:
PYBIND11_MODULE(AlphaZeroCpp, m) {
    m.doc() = "pybind11 bindings for custom MCTS + inference client";

    init();

    bind_go_game(m);

    py::class_<MCTSChild>(m, "MCTSChild")
        .def_readonly("move", &MCTSChild::move)
        .def_readonly("encoded_move", &MCTSChild::encoded_move)
        .def_readonly("raw_policy", &MCTSChild::raw_policy)
        .def_readonly("policy", &MCTSChild::policy)
        .def_readonly("visits", &MCTSChild::visits)
        .def_readonly("result_sum", &MCTSChild::result_sum)
        .def_readonly("virtual_loss", &MCTSChild::virtual_loss)
        .def_readonly("is_materialized", &MCTSChild::is_materialized);

    py::class_<MCTSRoot>(m, "MCTSRoot")
        .def_property_readonly("fen", [](const MCTSRoot &root) { return root.board().fen(); })
        .def_property_readonly("move", &MCTSRoot::move)
        .def_property_readonly("visits", &MCTSRoot::visits)
        .def_property_readonly("virtual_loss", &MCTSRoot::virtualLoss)
        .def_property_readonly("result_sum", &MCTSRoot::resultSum)
        .def_property_readonly("is_terminal", &MCTSRoot::isTerminal)
        .def_property_readonly("repetition_count",
                               [](const MCTSRoot &root) { return root.board().repetitionCount(); })
        .def_property_readonly("is_expanded", &MCTSRoot::isExpanded)
        .def_property_readonly("max_depth", &MCTSRoot::maxDepth)
        .def_property_readonly("children", &MCTSRoot::children)
        .def_property_readonly("live_nodes", &MCTSRoot::liveNodeCount)
        .def_property_readonly("total_child_records", &MCTSRoot::totalChildCount)
        .def_property_readonly("arena_capacity", &MCTSRoot::arenaCapacity)
        .def("make_new_root", &MCTSRoot::makeNewRoot, py::arg("child_index"),
             R"pbdoc(
            Prune the old tree and return a new root node.
            `child_index` is the index of the child to make the new root.
            )pbdoc")
        .def(
            "reset", &MCTSRoot::reset,
            R"pbdoc(Discard logical search state while retaining reusable arena allocations.)pbdoc")
        .def("discount", &MCTSRoot::discount, py::arg("percentage_of_node_visits_to_keep"),
             R"pbdoc(
            Discount the node's score and visits by a percentage.
            Descendant materializations are explicitly pruned when required by the fixed arena.
            )pbdoc")
        .def("__repr__", &MCTSRoot::repr);

    py::class_<MCTSBoard>(m, "MCTSBoard")
        .def(py::init<MCTSRoot, bool>(), py::arg("root"), py::arg("should_run_full_search"))
        .def_readonly("root", &MCTSBoard::root)
        .def_readonly("should_run_full_search", &MCTSBoard::should_run_full_search);

    m.def(
        "new_root",
        [](const std::string &fen, const uint32 arenaCapacity) {
            return MCTSRoot::create(fen, arenaCapacity);
        },
        py::arg("fen"), py::arg("arena_capacity"),
        R"pbdoc(
            Create a self-play MCTS root with an explicit fixed arena capacity.
          )pbdoc");
    m.def(
        "new_root_with_history",
        [](const std::string &startingFen, const std::vector<std::string> &movesUci,
           const uint32 arenaCapacity) {
            return MCTSRoot::create(ChessGameContract::replayPosition(startingFen, movesUci),
                                    arenaCapacity);
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

    // --- (2.1) MCTSParams ---
    py::class_<MCTSParams>(m, "MCTSParams")
        .def(py::init<int, uint32, uint32, float, float, float, uint8, uint8>(),
             py::arg("num_parallel_searches"), py::arg("num_full_searches"),
             py::arg("num_fast_searches"), py::arg("c_param"), py::arg("dirichlet_alpha"),
             py::arg("dirichlet_epsilon"), py::arg("min_visit_count"), py::arg("num_threads"))
        .def_readwrite("num_parallel_searches", &MCTSParams::num_parallel_searches)
        .def_readwrite("num_full_searches", &MCTSParams::num_full_searches)
        .def_readwrite("num_fast_searches", &MCTSParams::num_fast_searches)
        .def_readwrite("c_param", &MCTSParams::c_param)
        .def_readwrite("dirichlet_alpha", &MCTSParams::dirichlet_alpha)
        .def_readwrite("dirichlet_epsilon", &MCTSParams::dirichlet_epsilon)
        .def_readwrite("min_visit_count", &MCTSParams::min_visit_count)
        .def_readwrite("num_threads", &MCTSParams::num_threads);

    // --- (2.2) InferenceClientParams ---
    py::enum_<InferenceDevice>(m, "InferenceDevice")
        .value("AUTO", InferenceDevice::Auto)
        .value("CPU", InferenceDevice::Cpu)
        .value("CUDA", InferenceDevice::Cuda);

    py::class_<InferenceClientParams>(m, "InferenceClientParams")
        .def(py::init<int, std::string, int, int>(), py::arg("device_id"),
             py::arg("currentModelPath"), py::arg("maxBatchSize"),
             py::arg("microsecondsTimeoutInferenceThread"))
        .def(py::init<int, std::string, int, int, InferenceDevice>(), py::arg("device_id"),
             py::arg("currentModelPath"), py::arg("maxBatchSize"),
             py::arg("microsecondsTimeoutInferenceThread"), py::arg("device"))
        .def_readwrite("device_id", &InferenceClientParams::device_id)
        .def_readwrite("currentModelPath", &InferenceClientParams::currentModelPath)
        .def_readwrite("maxBatchSize", &InferenceClientParams::maxBatchSize)
        .def_readwrite("microsecondsTimeoutInferenceThread",
                       &InferenceClientParams::microsecondsTimeoutInferenceThread,
                       R"pbdoc(
                Timeout for the inference thread in microseconds.
                Default is 500 microseconds.
            )pbdoc")
        .def_readwrite("device", &InferenceClientParams::device);

    // --- (2.3) InferenceStatistics ---
    py::class_<InferenceStatistics>(m, "InferenceStatistics")
        .def(py::init<>())
        .def_readonly("evaluations", &InferenceStatistics::evaluations)
        .def_readonly("modelInferenceCalls", &InferenceStatistics::modelInferenceCalls)
        .def_readonly("modelInferencePositions", &InferenceStatistics::modelInferencePositions)
        .def_readonly("modelBatchSizeHistogram", &InferenceStatistics::modelBatchSizeHistogram)
        .def_readonly("averageNumberOfPositionsInInferenceCall",
                      &InferenceStatistics::averageNumberOfPositionsInInferenceCall)
        .def_readonly("treeSelectionNanoseconds", &InferenceStatistics::treeSelectionNanoseconds)
        .def_readonly("boardEncodingNanoseconds", &InferenceStatistics::boardEncodingNanoseconds)
        .def_readonly("resultProcessingNanoseconds",
                      &InferenceStatistics::resultProcessingNanoseconds)
        .def_readonly("treeBackupNanoseconds", &InferenceStatistics::treeBackupNanoseconds)
        .def_readonly("treeOwnerWaitNanoseconds", &InferenceStatistics::treeOwnerWaitNanoseconds)
        .def_readonly("directInferenceNanoseconds",
                      &InferenceStatistics::directInferenceNanoseconds)
        .def_readonly("directWorkerUtilization", &InferenceStatistics::directWorkerUtilization);

    // --- (2.4) MCTSResult ---
    py::class_<MCTSResult>(m, "MCTSResult")
        .def_readonly("result", &MCTSResult::result)
        .def_readonly("visits", &MCTSResult::visits) // vector<tuple<string,int>>
        .def_readonly("root", &MCTSResult::root);    // PyMCTSNode, the root node of the search tree

    // --- (2.5) MCTSStatistics ---
    py::class_<MCTSStatistics>(m, "MCTSStatistics")
        .def_readonly("averageDepth", &MCTSStatistics::averageDepth)
        .def_readonly("averageEntropy", &MCTSStatistics::averageEntropy)
        .def_readonly("averageKLDivergence", &MCTSStatistics::averageKLDivergence)
        .def_readonly("averagePolicySearchKLDivergence",
                      &MCTSStatistics::averagePolicySearchKLDivergence)
        .def_readonly("topMoveDisagreement", &MCTSStatistics::topMoveDisagreement)
        .def_readonly("selectedMovePriorRank", &MCTSStatistics::selectedMovePriorRank);

    // --- (2.6) MCTSResults ---
    py::class_<MCTSResults>(m, "MCTSResults")
        .def_readonly("results", &MCTSResults::results)     // vector<PyMCTSResult>
        .def_readonly("mctsStats", &MCTSResults::mctsStats) // PyMCTSStatistics
        .def_readonly("searchesCompleted", &MCTSResults::searchesCompleted);

    py::class_<DirectSelfPlayInferenceParams>(m, "DirectSelfPlayInferenceParams")
        .def(py::init<int, int, int>(), py::arg("inference_workers"),
             py::arg("inference_batch_size"), py::arg("outstanding_batches_per_worker") = 2)
        .def_readwrite("inference_workers", &DirectSelfPlayInferenceParams::inference_workers)
        .def_readwrite("inference_batch_size", &DirectSelfPlayInferenceParams::inference_batch_size)
        .def_readwrite("outstanding_batches_per_worker",
                       &DirectSelfPlayInferenceParams::outstanding_batches_per_worker);

    py::class_<FunctionTimeInfo>(m, "FunctionTimeInfo")
        .def_readonly("name", &FunctionTimeInfo::name)
        .def_readonly("percent", &FunctionTimeInfo::percent)
        .def_readonly("total", &FunctionTimeInfo::total)
        .def_readonly("invocations", &FunctionTimeInfo::invocations);

    py::class_<TimeInfo>(m, "TimeInfo")
        .def_readonly("totalTime", &TimeInfo::totalTime)
        .def_readonly("percentRecorded", &TimeInfo::percentRecorded)
        .def_readonly("functionTimes", &TimeInfo::functionTimes);

    // --- (4) MCTS class itself ---
    py::class_<MCTS>(m, "MCTS")
        .def(py::init<const InferenceClientParams &, const MCTSParams &,
                      std::optional<DirectSelfPlayInferenceParams>, uint64>(),
             py::arg("client_args"), py::arg("mcts_args"),
             py::arg("direct_inference_params") = std::nullopt,
             py::arg("initial_model_version") = 0)
        .def_property_readonly("arena_capacity", &MCTS::arenaCapacity)
        .def_property_readonly("model_version", &MCTS::modelVersion)
        .def(
            "new_root", [](const MCTS &self, const std::string &fen) { return self.newRoot(fen); },
            py::arg("fen"))
        .def(
            "new_root_with_history",
            [](const MCTS &self, const std::string &startingFen,
               const std::vector<std::string> &movesUci) {
                return self.newRoot(ChessGameContract::replayPosition(startingFen, movesUci));
            },
            py::arg("starting_fen"), py::arg("moves_uci"))
        .def("get_inference_statistics", &MCTS::getInferenceStatistics)
        .def("refresh_model", &MCTS::refreshModel, py::arg("model_version"), py::arg("model_path"),
             py::call_guard<py::gil_scoped_release>())
        .def("update_search_schedule", &MCTS::updateSearchSchedule, py::arg("mcts_args"))
        .def("search", &MCTS::search, py::arg("boards"), py::arg("collect_statistics") = false,
             R"pbdoc(
                 Run MCTS search on a list of boards.
                 `boards` should be a list of MCTSBoard values.
                 Returns an `MCTSResults` object, whose `.results` is a list of `MCTSResult`:
                     - result: float
                     - visits: List of (encoded_move: int, visit_count: int)
                     - children: List of NodeId (uint32)
                 When `collect_statistics` is true, `.mctsStats` contains
                 depth/entropy/KL for one representative root.
             )pbdoc")
        .def("inference", &inference, py::arg("fen"),
             R"pbdoc(
                 Run inference on a given FEN string.
                 Returns a tuple of (encoded_moves: List[Tuple[int, float]], value: float).
                 The encoded moves are pairs of (encoded_move: int, score: float).
             )pbdoc");

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
        .def(py::init<const InferenceClientParams &, const InteractiveSearchParams &>(),
             py::arg("client_parameters"), py::arg("search_parameters"))
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
