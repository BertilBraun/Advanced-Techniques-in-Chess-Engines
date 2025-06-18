#include "common.hpp"

#include "InferenceClient.hpp"
#include "MCTS/MCTS.hpp"
#include "MoveEncoding.hpp"

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
}

void testInferenceSpeed(int numBoards, int numIterations) {
    const InferenceClientParams params(0, "training_data/chess/model_0.pt", numBoards);

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

// -----------------------------------------------------------------------------
// Thin Python handle giving *read‑only* access to a single node.
// -----------------------------------------------------------------------------
class PyMCTSNode {
public:
    PyMCTSNode(NodePool *pool, const NodeId id) : m_pool(pool), m_id(id) {
        if (!m_pool)
            throw std::runtime_error("NodePool* is null");
        m_node = m_pool->get(m_id);
        if (!m_node)
            throw std::runtime_error("NodeId " + std::to_string(id) + " is not live");
    }

    // ------------------------------ properties ------------------------------
    std::string fen() const { return m_node->board.fen(); }
    std::string move() const { return toString(m_node->move_to_get_here); }
    int visits() const { return m_node->number_of_visits; }
    float virtual_loss() const { return m_node->virtual_loss; }
    float result() const { return m_node->result_score; }
    float policy() const { return m_node->policy; }
    float ucb(const float cParam) const {
        const MCTSNode *parent = m_pool->get(m_node->parent);
        return m_node->ucb(cParam * std::sqrt(parent->number_of_visits),
                           parent->result_score / static_cast<float>(parent->number_of_visits));
    }
    bool is_terminal() const { return m_node->isTerminalNode(); }
    bool is_fully_expanded() const { return m_node->isFullyExpanded(); }

    // ------------------------------ relations -------------------------------
    py::object parent() const {
        if (m_node->parent == INVALID_NODE)
            return py::none();
        return py::cast(PyMCTSNode(m_pool, m_node->parent));
    }
    std::vector<PyMCTSNode> children() const {
        std::vector<PyMCTSNode> out;
        out.reserve(m_node->children.size());
        for (NodeId cid : m_node->children)
            out.emplace_back(m_pool, cid);
        return out;
    }
    NodeId id() const { return m_id; }

    std::string repr() const { return m_node->repr(); }

private:
    NodePool *m_pool; // non‑owning – must outlive this handle
    NodeId m_id;
    MCTSNode *m_node; // cached for speed
};

PyMCTSNode get_node(MCTS &self, const NodeId id) {
    std::cout << "get_node(" << id << ")" << std::endl;
    return PyMCTSNode(self.getNodePool(), id);
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
        .def(py::init<int, float, float, float, float, uint8, uint8, uint32, uint32>(),
             py::arg("num_parallel_searches"), py::arg("c_param"), py::arg("dirichlet_alpha"),
             py::arg("dirichlet_epsilon"), py::arg("node_reuse_discount"),
             py::arg("min_visit_count"), py::arg("num_threads"), py::arg("num_full_searches"),
             py::arg("num_fast_searches"))
        .def_readwrite("num_parallel_searches", &MCTSParams::num_parallel_searches)
        .def_readwrite("c_param", &MCTSParams::c_param)
        .def_readwrite("dirichlet_alpha", &MCTSParams::dirichlet_alpha)
        .def_readwrite("dirichlet_epsilon", &MCTSParams::dirichlet_epsilon)
        .def_readwrite("node_reuse_discount", &MCTSParams::node_reuse_discount)
        .def_readwrite("min_visit_count", &MCTSParams::min_visit_count)
        .def_readwrite("num_threads", &MCTSParams::num_threads)
        .def_readwrite("num_full_searches", &MCTSParams::num_full_searches)
        .def_readwrite("num_fast_searches", &MCTSParams::num_fast_searches);

    // --- (2.2) InferenceClientParams ---
    py::class_<InferenceClientParams>(m, "InferenceClientParams")
        .def(py::init<int, std::string, int>(), py::arg("device_id"), py::arg("currentModelPath"),
             py::arg("maxBatchSize"))
        .def_readwrite("device_id", &InferenceClientParams::device_id)
        .def_readwrite("currentModelPath", &InferenceClientParams::currentModelPath)
        .def_readwrite("maxBatchSize", &InferenceClientParams::maxBatchSize);

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
        .def_readonly("visits", &MCTSResult::visits)      // vector<tuple<string,int>>
        .def_readonly("children", &MCTSResult::children); // vector<NodeId>

    // --- (2.5) MCTSStatistics ---
    py::class_<MCTSStatistics>(m, "MCTSStatistics")
        .def_readonly("averageDepth", &MCTSStatistics::averageDepth)
        .def_readonly("averageEntropy", &MCTSStatistics::averageEntropy)
        .def_readonly("averageKLDivergence", &MCTSStatistics::averageKLDivergence)
        .def_readonly("nodePoolCapacity", &MCTSStatistics::nodePoolCapacity)
        .def_readonly("liveNodeCount", &MCTSStatistics::liveNodeCount);

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
        .def("get_node", &get_node, py::arg("id"), R"pbdoc(
            Get a read-only handle to a specific MCTS node by its ID.
            Returns a `MCTSNode` object.
        )pbdoc")
        .def("clear_node_pool", &MCTS::clearNodePool,
             R"pbdoc(
                 Clear the MCTS node pool, releasing all nodes.
             )pbdoc")
        .def("inference", &inference, py::arg("fen"),
             R"pbdoc(
                 Run inference on a given FEN string.
                 Returns a tuple of (encoded_moves: List[Tuple[int, float]], value: float).
                 The encoded moves are pairs of (encoded_move: int, score: float).
             )pbdoc")
        .def("free_tree", &MCTS::freeTree, py::arg("nodeId"), py::arg("excluded") = INVALID_NODE,
             R"pbdoc(
                 Free the MCTS tree starting from the given node ID.
                 This will deallocate the node and all its children.
             )pbdoc")
        .def("eval_search", &MCTS::evalSearch, py::arg("fen"), py::arg("prevNodeId"),
             py::arg("numberOfSearches"),
             R"pbdoc(
                 Evaluate a search starting from the given FEN string.
                 Returns a `MCTSResult` object containing the average result score and visit counts.
             )pbdoc");

    // set NodeId type
    m.attr("NodeId") = py::int_(INVALID_NODE); // Use int type for NodeId
    m.attr("INVALID_NODE") = INVALID_NODE;

    py::class_<PyMCTSNode>(m, "MCTSNode")
        .def_property_readonly("id", &PyMCTSNode::id)
        .def_property_readonly("parent", &PyMCTSNode::parent)
        .def_property_readonly("children", &PyMCTSNode::children)
        .def_property_readonly("fen", &PyMCTSNode::fen)
        .def_property_readonly("move", &PyMCTSNode::move)
        .def_property_readonly("visits", &PyMCTSNode::visits)
        .def_property_readonly("virtual_loss", &PyMCTSNode::virtual_loss)
        .def_property_readonly("result", &PyMCTSNode::result)
        .def_property_readonly("policy", &PyMCTSNode::policy)
        .def("ucb", &PyMCTSNode::ucb, py::arg("cParam"),
             R"pbdoc(
                 Calculate the UCB score for this node. `cParam` is the exploration parameter.
             )pbdoc")
        .def_property_readonly("is_terminal", &PyMCTSNode::is_terminal,
                               R"pbdoc(
                                 Check if this node is a terminal node (game over).
                             )pbdoc")
        .def_property_readonly("is_fully_expanded", &PyMCTSNode::is_fully_expanded,
                               R"pbdoc(
                                 Check if this node is fully expanded (has children).
                             )pbdoc")
        .def("__repr__", &PyMCTSNode::repr);
}
