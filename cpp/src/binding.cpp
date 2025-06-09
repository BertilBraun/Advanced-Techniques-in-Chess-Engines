#include "common.hpp"

#include "InferenceClient.hpp"
#include "MCTS/MCTS.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

static void init() {
    // We need to initialize the Stockfish engine before using it.
    Bitboards::init();
    Position::init();
}

// ——————————————————————————————————————————————
// Bind everything with pybind11:
PYBIND11_MODULE(AlphaZeroCpp, m) {
    m.doc() = "pybind11 bindings for custom MCTS + inference client";

    init(); // Initialize Stockfish engine

    // --- (2.1) MCTSParams ---
    py::class_<MCTSParams>(m, "MCTSParams")
        .def(py::init<int, float, float, float, float, uint8, uint8>(),
             py::arg("num_parallel_searches"), py::arg("c_param"), py::arg("dirichlet_alpha"),
             py::arg("dirichlet_epsilon"), py::arg("node_reuse_discount"),
             py::arg("min_visit_count"), py::arg("num_threads"))
        .def_readwrite("num_parallel_searches", &MCTSParams::num_parallel_searches)
        .def_readwrite("c_param", &MCTSParams::c_param)
        .def_readwrite("dirichlet_alpha", &MCTSParams::dirichlet_alpha)
        .def_readwrite("dirichlet_epsilon", &MCTSParams::dirichlet_epsilon)
        .def_readwrite("node_reuse_discount", &MCTSParams::node_reuse_discount)
        .def_readwrite("min_visit_count", &MCTSParams::min_visit_count)
        .def_readwrite("num_threads", &MCTSParams::num_threads);

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
        .def_readonly("nnOutputValueDistribution", &InferenceStatistics::nnOutputValueDistribution);

    // --- (2.4) MCTSResult (Python view) ---
    py::class_<MCTSResult>(m, "MCTSResult")
        .def_readonly("result", &MCTSResult::result)
        .def_readonly("visits", &MCTSResult::visits)     // vector<tuple<string,int>>
        .def_readonly("children", &MCTSResult::children) // vector<NodeId>
        ;

    // --- (2.5) MCTSStatistics (same as C++) ---
    py::class_<MCTSStatistics>(m, "MCTSStatistics")
        .def_readonly("averageDepth", &MCTSStatistics::averageDepth)
        .def_readonly("averageEntropy", &MCTSStatistics::averageEntropy)
        .def_readonly("averageKLDivergence", &MCTSStatistics::averageKLDivergence);

    // --- (2.6) MCTSResults (Python view) ---
    py::class_<MCTSResults>(m, "MCTSResults")
        .def_readonly("results", &MCTSResults::results)      // vector<PyMCTSResult>
        .def_readonly("mctsStats", &MCTSResults::mctsStats); // PyMCTSStatistics

    // --- (4) MCTS class itself ---
    py::class_<MCTS>(m, "MCTS")
        .def(py::init<const InferenceClientParams &, const MCTSParams &>(), py::arg("client_args"),
             py::arg("mcts_args"))
        .def("get_inference_statistics", &MCTS::getInferenceStatistics)
        .def("search", &MCTS::search, py::arg("boards"),
             R"pbdoc(
                 Run MCTS search on a list of boards.
                 `boards` should be a list of tuples: (fen_str: str, prev_node: int, num_searches: int).
                 Returns an `MCTSResults` object, whose `.results` is a list of `MCTSResult`:
                     - result: float
                     - visits: List of (encoded_move: int, visit_count: int)
                     - children: List of NodeId (uint32)
                 and `.mctsStats` contains avg depth/entropy/KL.
             )pbdoc");

    m.attr("INVALID_NODE") = INVALID_NODE;
}
