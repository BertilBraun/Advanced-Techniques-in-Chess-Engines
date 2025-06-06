#include "common.hpp"

#include "InferenceClient.hpp"
#include "MCTS/MCTS.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// ——————————————————————————————————————————————
// We create “Python‐friendly” mirror structs for anything that used `Move`.
//
struct PyMCTSResult {
    float result;
    std::vector<std::pair<std::string, int>> visits; // <uci_move, visit_count>
    std::vector<NodeId> children;
};

using PyMCTSStatistics = MCTSStatistics;

struct PyMCTSResults {
    std::vector<PyMCTSResult> results;
    PyMCTSStatistics mctsStats;
};

// ——————————————————————————————————————————————
// A little wrapper around your real MCTS::search(...) so we convert Move→string:
static PyMCTSResults
mcts_search_wrapper(MCTS &self,
                    const std::vector<std::tuple<std::string, NodeId, int>> &boards_in) {
    // Call the real C++ search:
    const auto [results, mctsStats] = self.search(boards_in);

    // Now convert to PyMCTSResults:
    PyMCTSResults pyOut;
    pyOut.mctsStats = mctsStats;

    for (const auto &[result, visits, children] : results) {
        PyMCTSResult r2;
        r2.result = result;
        r2.children = children; // copy NodeId vector directly

        // Convert each (Move, int) → (uci_string, int)
        for (const auto &[move, cnt] : visits) {
            std::string uci = toString(move); // Convert Move to UCI string
            r2.visits.emplace_back(uci, cnt);
        }
        pyOut.results.push_back(std::move(r2));
    }
    return pyOut;
}

// ——————————————————————————————————————————————
// Bind everything with pybind11:
PYBIND11_MODULE(AlphaZeroCpp, m) {
    m.doc() = "pybind11 bindings for custom MCTS + inference client";

    // --- (2.1) MCTSParams ---
    py::class_<MCTSParams>(m, "MCTSParams")
        .def(py::init<>())
        .def_readwrite("num_parallel_searches", &MCTSParams::num_parallel_searches)
        .def_readwrite("c_param", &MCTSParams::c_param)
        .def_readwrite("dirichlet_alpha", &MCTSParams::dirichlet_alpha)
        .def_readwrite("dirichlet_epsilon", &MCTSParams::dirichlet_epsilon)
        .def_readwrite("node_reuse_discount", &MCTSParams::node_reuse_discount)
        .def_readwrite("min_visit_count", &MCTSParams::min_visit_count);

    // --- (2.2) InferenceClientParams ---
    py::class_<InferenceClientParams>(m, "InferenceClientParams")
        .def(py::init<>())
        .def_readwrite("device_id", &InferenceClientParams::device_id)
        .def_readwrite("currentModelPath", &InferenceClientParams::currentModelPath)
        .def_readwrite("maxBatchSize", &InferenceClientParams::maxBatchSize);

    // --- (2.3) InferenceStatistics ---
    py::class_<InferenceStatistics>(m, "InferenceStatistics")
        .def(py::init<>())
        .def_readwrite("cacheHitRate", &InferenceStatistics::cacheHitRate)
        .def_readwrite("uniquePositions", &InferenceStatistics::uniquePositions)
        .def_readwrite("cacheSizeMB", &InferenceStatistics::cacheSizeMB)
        .def_readwrite("nnOutputValueDistribution",
                       &InferenceStatistics::nnOutputValueDistribution);

    // --- (2.4) MCTSResult (Python view) ---
    py::class_<PyMCTSResult>(m, "MCTSResult")
        .def_readonly("result", &PyMCTSResult::result)
        .def_readonly("visits", &PyMCTSResult::visits)     // vector<tuple<string,int>>
        .def_readonly("children", &PyMCTSResult::children) // vector<NodeId>
        ;

    // --- (2.5) MCTSStatistics (same as C++) ---
    py::class_<PyMCTSStatistics>(m, "MCTSStatistics")
        .def_readonly("averageDepth", &PyMCTSStatistics::averageDepth)
        .def_readonly("averageEntropy", &PyMCTSStatistics::averageEntropy)
        .def_readonly("averageKLDivergence", &PyMCTSStatistics::averageKLDivergence);

    // --- (2.6) MCTSResults (Python view) ---
    py::class_<PyMCTSResults>(m, "MCTSResults")
        .def_readonly("results", &PyMCTSResults::results)     // vector<PyMCTSResult>
        .def_readonly("mctsStats", &PyMCTSResults::mctsStats) // PyMCTSStatistics
        ;

    // --- (4) MCTS class itself ---
    py::class_<MCTS>(m, "MCTS")
        .def(py::init<const InferenceClientParams &, const MCTSParams &>(), py::arg("client_args"),
             py::arg("mcts_args"))
        // Bind getInferenceStatistics() as a simple method:
        .def("get_inference_statistics", &MCTS::getInferenceStatistics)
        // Bind our wrapper of search(...) that converts Move→string
        .def("search", &mcts_search_wrapper, py::arg("boards"),
             R"pbdoc(
                 Run MCTS search on a list of boards.
                 `boards` should be a list of tuples: (fen_str: str, prev_node: int, num_searches: int).
                 Returns an `MCTSResults` object, whose `.results` is a list of `MCTSResult`:
                     - result: float
                     - visits: List of (uci_move: str, visit_count: int)
                     - children: List of NodeId (uint32)
                 and `.mctsStats` contains avg depth/entropy/KL.
             )pbdoc");

    m.attr("INVALID_NODE") = INVALID_NODE;
}
