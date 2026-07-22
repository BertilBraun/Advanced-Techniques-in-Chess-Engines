#include "common.hpp"

#include "BoardEncoding.hpp"
#include "InferenceClient.hpp"
#include "InteractiveEngine.hpp"
#include "MCTS/MCTS.hpp"
#include "MoveEncoding.hpp"

#include "MCTS/EvalMCTS.hpp"

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

Board *newBoard() {
    Board *board = new Board();
    for (int i = 0; i < 30; ++i) {
        const std::vector<Move> &moves = board->validMoves();
        if (moves.empty())
            break; // No more valid moves, stop early
        board->makeMove(moves[rand() % moves.size()]);
    }
    if (board->isGameOver()) {
        delete board;      // Clean up if the game is over
        return newBoard(); // Create a new board if the game is over
    }
    return board;
}

std::vector<const Board *> newBoards(const int numBoards) {
    std::vector<const Board *> boards;
    boards.reserve(numBoards);
    for (int _ : range(numBoards))
        boards.push_back(newBoard());

    return boards;
}

void testInferenceSpeed(int numBoards, int numIterations) {
    const InferenceClientParams params(0, "training_data/chess/model_0.pt", numBoards, 100, 100000);

    InferenceClient client(params);

    float totalTime = 0.0f;
    for (int i = 0; i < numIterations; ++i) {
        std::vector<const Board *> boards = newBoards(numBoards);

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<InferenceResult> results = client.inferenceBatch(boards);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = end - start;
        totalTime += duration.count();
        std::cout << "Iteration " << i + 1 << ": Inference time: " << duration.count()
                  << " seconds\n";
        for (const Board *board : boards) {
            delete board; // Clean up the boards
        }
    }

    std::cout << "Total time: " << totalTime << " seconds\n";
    std::cout << "Average time per iteration: " << (totalTime / numIterations) << " seconds\n";
    std::cout << "Average time per board: " << (totalTime / (numIterations * numBoards))
              << " seconds\n";
    std::cout << std::endl;

    resetTimes();
}

void testMCTSSpeed(int numBoards, int numIterations, int numSearchesPerTurn,
                   int numParallelSearches, int numThreads) {
    const MCTSParams mctsParams(numParallelSearches, numSearchesPerTurn, numSearchesPerTurn, 1.0,
                                0.3, 0.0, 0, numThreads);

    const InferenceClientParams inferenceParams(0, "training_data/chess/model_0.pt",
                                                numBoards * numParallelSearches, 100, 100000);

    MCTS mcts(inferenceParams, mctsParams);

    float totalTime = 0.0f;

    for (int i = 0; i < numIterations; ++i) {
        std::vector<MCTSBoard> boards;
        boards.reserve(numBoards);
        for (auto &&board : newBoards(numBoards)) {
            boards.emplace_back(mcts.newRoot(board->fen()), true);
        }

        auto start = std::chrono::high_resolution_clock::now();
        auto _ = mcts.search(boards);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = end - start;

        std::cout << "Iteration " << i + 1 << ": MCTS search time: " << duration.count()
                  << " seconds\n";

        totalTime += duration.count();
    }

    std::cout << "Total MCTS time: " << totalTime << " seconds\n";
    std::cout << "Average MCTS time per iteration: " << (totalTime / numIterations) << " seconds\n";
    std::cout << "Average MCTS time per board: " << (totalTime / (numIterations * numBoards))
              << " seconds\n";
}

void testEvalMCTSSpeed(int numBoards, int numIterations, int numSearchesPerTurn,
                       int numParallelSearches, int numThreads) {
    const EvalMCTSParams mctsParams(1.0, numThreads);

    const InferenceClientParams inferenceParams(0, "training_data/chess/model_0.pt",
                                                numBoards * numParallelSearches, 100, 100000);

    EvalMCTS mcts(inferenceParams, mctsParams);

    float totalTime = 0.0f;

    for (int i = 0; i < numIterations; ++i) {
        std::vector<std::shared_ptr<EvalMCTSNode>> roots;
        roots.reserve(numBoards);
        for (auto &&board : newBoards(numBoards)) {
            // Create a root node for each board
            roots.emplace_back(EvalMCTSNode::createRoot(board->fen()));
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (const auto &root : roots) {
            // Run Eval MCTS search on each root node
            auto _ = mcts.evalSearch(root, numSearchesPerTurn);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = end - start;

        std::cout << "Iteration " << i + 1 << ": Eval MCTS search time: " << duration.count()
                  << " seconds\n";

        totalTime += duration.count();
    }

    std::cout << "Total Eval MCTS time: " << totalTime << " seconds\n";
    std::cout << "Average Eval MCTS time per iteration: " << (totalTime / numIterations)
              << " seconds\n";
    std::cout << "Average Eval MCTS time per board: " << (totalTime / (numIterations * numBoards))
              << " seconds\n";
}

std::pair<std::vector<std::pair<int, float>>, float> inference(MCTS &self, const std::string &fen) {
    const Board board(fen);

    std::vector<const Board *> boards;
    boards.push_back(&board);

    const auto result = self.inferenceBatch(boards);
    assert(result.size() == 1 && "Inference should return exactly one result for one board");
    const InferenceResult &inferenceResult = result[0];

    std::vector<std::pair<int, float>> encodedMoves;
    encodedMoves.reserve(inferenceResult.moves.size());
    for (const auto &[move, score] : inferenceResult.moves) {
        encodedMoves.emplace_back(encodeMove(move, &board), score);
    }

    return {encodedMoves, inferenceResult.value()};
}

// ——————————————————————————————————————————————
// Bind everything with pybind11:
PYBIND11_MODULE(AlphaZeroCpp, m) {
    m.doc() = "pybind11 bindings for custom MCTS + inference client";

    init();

    py::class_<MCTSChild>(m, "MCTSChild")
        .def_readonly("move", &MCTSChild::move)
        .def_readonly("encoded_move", &MCTSChild::encoded_move)
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
            return MCTSRoot::create(replayMoves(startingFen, movesUci), arenaCapacity);
        },
        py::arg("starting_fen"), py::arg("moves_uci"), py::arg("arena_capacity"),
        R"pbdoc(Create a fixed-capacity MCTS root by replaying bounded UCI history.)pbdoc");

    m.def(
        "encode_board_compressed",
        [](const std::string &fen)
            -> std::pair<std::array<uint64, BINARY_C>, std::array<int8, SCALAR_C>> {
            const Board board(fen);
            const CompressedEncodedBoard encoded = encodeBoard(&board);
            return {encoded.bits, encoded.scal};
        },
        py::arg("fen"),
        R"pbdoc(Encode a FEN into the canonical compressed binary and scalar planes.)pbdoc");

    m.def("test_inference_speed_cpp", &testInferenceSpeed,
          "Test the inference speed of the InferenceClient", py::arg("numBoards") = 100,
          py::arg("numIterations") = 10,
          R"pbdoc(
            Test the inference speed of the InferenceClient.
            Runs inference on a specified number of boards for a given number of iterations.
            Prints the average time taken per iteration and per board.
          )pbdoc");

    m.def("test_mcts_speed_cpp", &testMCTSSpeed, "Test the MCTS search speed",
          py::arg("numBoards") = 100, py::arg("numIterations") = 10,
          py::arg("numSearchesPerTurn") = 100, py::arg("numParallelSearches") = 1,
          py::arg("numThreads") = 1,
          R"pbdoc(
            Test the MCTS search speed.
            Runs MCTS search on a specified number of boards for a given number of iterations.
            Prints the average time taken per iteration and per board.
          )pbdoc");

    m.def("test_eval_mcts_speed_cpp", &testEvalMCTSSpeed, "Test the Eval MCTS search speed",
          py::arg("numBoards") = 100, py::arg("numIterations") = 10,
          py::arg("numSearchesPerTurn") = 100, py::arg("numParallelSearches") = 1,
          py::arg("numThreads") = 1,
          R"pbdoc(
            Test the Eval MCTS search speed.
            Runs Eval MCTS search on a specified number of boards for a given number of iterations.
            Prints the average time taken per iteration and per board.
          )pbdoc");

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
        .def(py::init<int, std::string, int, int, size_t>(), py::arg("device_id"),
             py::arg("currentModelPath"), py::arg("maxBatchSize"),
             py::arg("microsecondsTimeoutInferenceThread"), py::arg("cacheCapacity"))
        .def(py::init<int, std::string, int, int, size_t, InferenceDevice>(), py::arg("device_id"),
             py::arg("currentModelPath"), py::arg("maxBatchSize"),
             py::arg("microsecondsTimeoutInferenceThread"), py::arg("cacheCapacity"),
             py::arg("device"))
        .def_readwrite("device_id", &InferenceClientParams::device_id)
        .def_readwrite("currentModelPath", &InferenceClientParams::currentModelPath)
        .def_readwrite("maxBatchSize", &InferenceClientParams::maxBatchSize)
        .def_readwrite("microsecondsTimeoutInferenceThread",
                       &InferenceClientParams::microsecondsTimeoutInferenceThread,
                       R"pbdoc(
                Timeout for the inference thread in microseconds.
                Default is 500 microseconds.
            )pbdoc")
        .def_readwrite("cacheCapacity", &InferenceClientParams::cacheCapacity)
        .def_readwrite("device", &InferenceClientParams::device);

    // --- (2.3) InferenceStatistics ---
    py::class_<InferenceStatistics>(m, "InferenceStatistics")
        .def(py::init<>())
        .def_readonly("cacheHitRate", &InferenceStatistics::cacheHitRate)
        .def_readonly("evaluations", &InferenceStatistics::evaluations)
        .def_readonly("cacheHits", &InferenceStatistics::cacheHits)
        .def_readonly("uniquePositions", &InferenceStatistics::uniquePositions)
        .def_readonly("cacheSizeMB", &InferenceStatistics::cacheSizeMB)
        .def_readonly("cacheCapacity", &InferenceStatistics::cacheCapacity)
        .def_readonly("cacheEvictions", &InferenceStatistics::cacheEvictions)
        .def_readonly("cacheFingerprintCollisions",
                      &InferenceStatistics::cacheFingerprintCollisions)
        .def_readonly("nnOutputValueDistribution", &InferenceStatistics::nnOutputValueDistribution)
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
        .def_readonly("averageKLDivergence", &MCTSStatistics::averageKLDivergence);

    // --- (2.6) MCTSResults ---
    py::class_<MCTSResults>(m, "MCTSResults")
        .def_readonly("results", &MCTSResults::results)      // vector<PyMCTSResult>
        .def_readonly("mctsStats", &MCTSResults::mctsStats); // PyMCTSStatistics

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
        .def(py::init<const InferenceClientParams &, const MCTSParams &, bool>(),
             py::arg("client_args"), py::arg("mcts_args"), py::arg("use_inference_cache") = true)
        .def_property_readonly("arena_capacity", &MCTS::arenaCapacity)
        .def(
            "new_root", [](const MCTS &self, const std::string &fen) { return self.newRoot(fen); },
            py::arg("fen"))
        .def(
            "new_root_with_history",
            [](const MCTS &self, const std::string &startingFen,
               const std::vector<std::string> &movesUci) {
                return self.newRoot(replayMoves(startingFen, movesUci));
            },
            py::arg("starting_fen"), py::arg("moves_uci"))
        .def("get_inference_statistics", &MCTS::getInferenceStatistics)
        .def("update", &MCTS::update, py::arg("model_path"), py::arg("mcts_args"))
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

    // Expose EvalMCTSNode and EvalMCTS
    py::class_<EvalMCTSResult>(m, "EvalMCTSResult")
        .def_readonly("result", &EvalMCTSResult::result)
        .def_readonly("visits", &EvalMCTSResult::visits)
        // EvalMCTSNode, the root node of the search tree
        .def_readonly("root", &EvalMCTSResult::root);

    py::class_<EvalMCTSParams>(m, "EvalMCTSParams")
        .def(py::init<float, uint8>(), py::arg("c_param"), py::arg("num_threads"))
        .def_readwrite("c_param", &EvalMCTSParams::c_param)
        .def_readwrite("num_threads", &EvalMCTSParams::num_threads);

    py::class_<EvalMCTSNode, std::shared_ptr<EvalMCTSNode>>(m, "EvalMCTSNode")
        .def_property_readonly("fen",
                               [](EvalMCTSNode &node) {
                                   node.materializeBoard();
                                   return node.board().fen();
                               })
        .def_property_readonly("is_terminal", &EvalMCTSNode::isTerminal)
        .def_property_readonly("repetition_count",
                               [](EvalMCTSNode &node) {
                                   node.materializeBoard();
                                   return node.board().repetitionCount();
                               })
        .def_property_readonly("children",
                               [](const EvalMCTSNode &node) {
                                   const auto children = node.children();
                                   return children == nullptr ? EvalMCTSNode::ChildVector{}
                                                              : *children;
                               })
        .def("best_child", &EvalMCTSNode::bestChild, py::arg("c_param"),
             R"pbdoc(
            Get the best child node based on UCB score.
            `c_param` is the exploration constant.
            )pbdoc")
        .def_property_readonly("visits",
                               [](const EvalMCTSNode &n) {
                                   return n.number_of_visits.load(std::memory_order_acquire);
                               })
        .def_property_readonly("move",
                               [](const EvalMCTSNode &n) { return toString(n.moveToGetHere); })
        .def_property_readonly("encoded_move",
                               [](const EvalMCTSNode &n) {
                                   return encodeMove(n.moveToGetHere, &n.parent.lock()->board());
                               })
        .def_property_readonly(
            "result_sum",
            [](const EvalMCTSNode &n) { return n.result_sum.load(std::memory_order_acquire); })
        .def_readonly("policy", &EvalMCTSNode::policy)
        .def_property_readonly("outcome_prediction", &EvalMCTSNode::outcomePrediction)
        .def_property_readonly("max_depth", &EvalMCTSNode::maxDepth)
        .def("make_new_root", &EvalMCTSNode::makeNewRoot, py::arg("child_index"),
             R"pbdoc(
            Prune the old tree and return a new root node.
            `child_index` is the index of the child to make the new root.
            )pbdoc");

    py::class_<EvalMCTS>(m, "EvalMCTS")
        .def(py::init<const InferenceClientParams &, const EvalMCTSParams &>(),
             py::arg("client_args"), py::arg("mcts_args"))
        .def("eval_search", &EvalMCTS::evalSearch, py::arg("root"), py::arg("searches"),
             R"pbdoc(
                 Run evaluation MCTS search on a given root node.
                 Returns an `EvalMCTSResult` object containing the result, visits, and root node.
             )pbdoc");

    m.def(
        "new_eval_root", [](const std::string &fen) { return EvalMCTSNode::createRoot(fen); },
        py::arg("fen"),
        R"pbdoc(
            Create a new root node for evaluation MCTS with the given FEN string.
            Returns a shared pointer to the new EvalMCTSNode.
          )pbdoc");
    m.def(
        "new_eval_root_with_history",
        [](const std::string &startingFen, const std::vector<std::string> &movesUci) {
            return EvalMCTSNode::createRoot(replayMoves(startingFen, movesUci));
        },
        py::arg("starting_fen"), py::arg("moves_uci"),
        R"pbdoc(Create an evaluation MCTS root by replaying a bounded UCI move history.)pbdoc");

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
             py::arg("time_limit_seconds") = std::nullopt, py::arg("search_limit") = std::nullopt)
        .def_property_readonly("fen", &InteractiveGame::fen)
        .def_property_readonly("starting_fen", &InteractiveGame::startingFen)
        .def_property_readonly("moves_uci", &InteractiveGame::movesUci)
        .def_property_readonly("root_visits", &InteractiveGame::rootVisits);
}
