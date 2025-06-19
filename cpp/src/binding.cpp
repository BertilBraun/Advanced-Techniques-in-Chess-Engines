#include "common.hpp"

#include "InferenceClient.hpp"
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

void testInferenceSpeed(int numBoards, int numIterations) {
    const InferenceClientParams params(0, "training_data/chess/model_0.pt", numBoards, 1000);

    InferenceClient client(params);

    float totalTime = 0.0f;
    for (int i = 0; i < numIterations; ++i) {
        std::vector<const Board *> boards;
        boards.reserve(numBoards);
        while (boards.size() < numBoards) {
            Board *board = new Board();
            for (int k = 0; k < 30; ++k) {
                auto moves = board->validMoves();
                if (moves.empty())
                    break; // No more valid moves, stop early
                board->makeMove(moves[rand() % moves.size()]);
            }
            if (board->isGameOver()) {
                delete board; // Clean up if the game is over
                continue;
            }
            boards.push_back(board);
        }

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

std::pair<std::vector<std::pair<int, float>>, float> inference(MCTS &self, const std::string &fen) {
    const Board board(fen);

    std::vector<const Board *> boards;
    boards.push_back(&board);

    const auto result = self.getInferenceClient()->inferenceBatch(boards);
    assert(result.size() == 1 && "Inference should return exactly one result for one board");
    const auto &[moves, value] = result[0];

    std::vector<std::pair<int, float>> encodedMoves;
    encodedMoves.reserve(moves.size());
    for (const auto &[move, score] : moves) {
        encodedMoves.emplace_back(encodeMove(move, &board), score);
    }

    return {encodedMoves, value};
}

// ——————————————————————————————————————————————
// Bind everything with pybind11:
PYBIND11_MODULE(AlphaZeroCpp, m) {
    m.doc() = "pybind11 bindings for custom MCTS + inference client";

    init();

    /**
     * NOTE - USAGE:
     * you want to searh a board FEN with MCTS:
     * 1. Create an MCTS instance with the desired parameters.
     * 2. Create a MCTSNode as the root node with the board FEN.
     * 3. Call the `search` method on the MCTS instance with a list of boards.
     * 4. Select the best move based on the visit counts from the MCTSResult.
     * 5. Select the child from the root node that corresponds to the best move.
     * 6. Call makeNewRoot on the MCTSNode to create a new root for the next search.
     * 7. Repeat steps 3-6 for subsequent searches with the new root node.
     */

    py::class_<MCTSNode, std::shared_ptr<MCTSNode>>(m, "MCTSNode")
        .def_property_readonly("fen", [](const MCTSNode &n) { return n.board.fen(); })
        .def_property_readonly("move",
                               [](const MCTSNode &n) { return toString(n.move_to_get_here); })
        .def_property_readonly(
            "encoded_move",
            [](const MCTSNode &n) { return encodeMove(n.move_to_get_here, &n.board); })
        .def_readonly("visits", &MCTSNode::number_of_visits)
        .def_readonly("virtual_loss", &MCTSNode::virtual_loss)
        .def_readonly("result_sum", &MCTSNode::result_sum)
        .def_readonly("policy", &MCTSNode::policy)
        .def(
            "ucb",
            [](const MCTSNode &n, const float cParam) {
                return n.ucb(cParam * std::sqrt(n.number_of_visits),
                             n.result_sum / static_cast<float>(n.number_of_visits));
            },
            py::arg("cParam"),
            R"pbdoc(Calculate UCB score given exploration constant cParam.)pbdoc")
        .def_property_readonly("is_terminal", &MCTSNode::isTerminal)
        .def_property_readonly("is_expanded", &MCTSNode::isExpanded)
        .def_property_readonly("max_depth", &MCTSNode::maxDepth)

        // Relations
        .def_property_readonly("parent",
                               [](const MCTSNode &node) {
                                   return node.parent.lock(); // returns shared_ptr or None
                               })
        .def_property_readonly(
            "children",
            [](const MCTSNode &node) {
                return node.children; // returns vector<shared_ptr<MCTSNode>>
            },
            py::return_value_policy::reference_internal)

        .def("best_child", &MCTSNode::bestChild, py::arg("cParam"))

        // Pruning and re-rooting
        .def("make_new_root", &MCTSNode::makeNewRoot, py::arg("child_index"),
             R"pbdoc(
            Prune the old tree and return a new root node.
            `child_index` is the index of the child to make the new root.
            )pbdoc")

        .def("discount", &MCTSNode::discount, py::arg("percentage_of_node_visits_to_keep"),
             R"pbdoc(
            Discount the node's score and visits by a percentage.
            This is useful for pruning old nodes in the search tree.
            )pbdoc")

        .def("__repr__", &MCTSNode::repr);

    // Root creation helper
    m.def("new_root", &MCTSNode::createRoot, py::arg("fen"),
          R"pbdoc(
            Create a new root node for MCTS with the given FEN string.
            Returns a shared pointer to the new MCTSNode.
          )pbdoc");

    m.def("test_inference_speed_cpp", &testInferenceSpeed,
          "Test the inference speed of the InferenceClient", py::arg("numBoards") = 100,
          py::arg("numIterations") = 10,
          R"pbdoc(
            Test the inference speed of the InferenceClient.
            Runs inference on a specified number of boards for a given number of iterations.
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
    py::class_<InferenceClientParams>(m, "InferenceClientParams")
        .def(py::init<int, std::string, int, int>(), py::arg("device_id"),
             py::arg("currentModelPath"), py::arg("maxBatchSize"),
             py::arg("microsecondsTimeoutInferenceThread") = 500)
        .def_readwrite("device_id", &InferenceClientParams::device_id)
        .def_readwrite("currentModelPath", &InferenceClientParams::currentModelPath)
        .def_readwrite("maxBatchSize", &InferenceClientParams::maxBatchSize)
        .def_readwrite("microsecondsTimeoutInferenceThread",
                       &InferenceClientParams::microsecondsTimeoutInferenceThread,
                       R"pbdoc(
                Timeout for the inference thread in microseconds.
                Default is 500 microseconds.
            )pbdoc");

    // --- (2.3) InferenceStatistics ---
    py::class_<InferenceStatistics>(m, "InferenceStatistics")
        .def(py::init<>())
        .def_readonly("cacheHitRate", &InferenceStatistics::cacheHitRate)
        .def_readonly("uniquePositions", &InferenceStatistics::uniquePositions)
        .def_readonly("cacheSizeMB", &InferenceStatistics::cacheSizeMB)
        .def_readonly("nnOutputValueDistribution", &InferenceStatistics::nnOutputValueDistribution)
        .def_readonly("averageNumberOfPositionsInInferenceCall",
                      &InferenceStatistics::averageNumberOfPositionsInInferenceCall);

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
        .def(py::init<const InferenceClientParams &, const MCTSParams &>(), py::arg("client_args"),
             py::arg("mcts_args"))
        .def("get_inference_statistics", &MCTS::getInferenceStatistics)
        .def("search", &MCTS::search, py::arg("boards"),
             R"pbdoc(
                 Run MCTS search on a list of boards.
                 `boards` should be a list of tuples: (fen_str: str, prev_node: NodeId, full_search: bool).
                 Returns an `MCTSResults` object, whose `.results` is a list of `MCTSResult`:
                     - result: float
                     - visits: List of (encoded_move: int, visit_count: int)
                     - children: List of NodeId (uint32)
                 and `.mctsStats` contains avg depth/entropy/KL.
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
        .def_property_readonly("fen", [](const EvalMCTSNode &n) { return n.board.fen(); })
        .def_property_readonly(
            "children",
            [](const EvalMCTSNode &n) {
                const auto children = n.childrenPtr.load(std::memory_order_acquire);
                return children ? *children : std::vector<std::shared_ptr<EvalMCTSNode>>{};
            },
            py::return_value_policy::reference_internal)
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
        .def_property_readonly(
            "encoded_move",
            [](const EvalMCTSNode &n) { return encodeMove(n.moveToGetHere, &n.board); })
        .def_property_readonly(
            "result_sum",
            [](const EvalMCTSNode &n) { return n.result_sum.load(std::memory_order_acquire); })
        .def_readonly("policy", &EvalMCTSNode::policy)
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

    m.def("new_eval_root", &EvalMCTSNode::createRoot, py::arg("fen"),
          R"pbdoc(
            Create a new root node for evaluation MCTS with the given FEN string.
            Returns a shared pointer to the new EvalMCTSNode.
          )pbdoc");
}
