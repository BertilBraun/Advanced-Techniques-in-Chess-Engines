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

// -----------------------------------------------------------------------------
// Thin Python handle giving *read‑only* access to a single node.
// -----------------------------------------------------------------------------
class PyMCTSNode {
public:
    PyMCTSNode(NodePool *pool, NodeId id) : m_pool(pool), m_id(id) {
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
    float ucb(float cParam, int parentNumberOfVisits) const {
        return m_node->ucb(cParam * std::sqrt(parentNumberOfVisits));
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

    std::string repr() const { return m_node->repr(); }

private:
    NodePool *m_pool; // non‑owning – must outlive this handle
    NodeId m_id;
    MCTSNode *m_node; // cached for speed
};

PyMCTSNode get_node(MCTS &self, NodeId id) { return PyMCTSNode(self.getNodePool(), id); }

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
        .def_readonly("averageKLDivergence", &MCTSStatistics::averageKLDivergence);

    // --- (2.6) MCTSResults ---
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
             )pbdoc")
        .def("get_node", &get_node, py::arg("id"), R"pbdoc(
            Get a read-only handle to a specific MCTS node by its ID.
            Returns a `MCTSNode` object.
        )pbdoc");

    // set NodeId type
    m.attr("NodeId") = py::int_(INVALID_NODE); // Use int type for NodeId
    m.attr("INVALID_NODE") = INVALID_NODE;

    py::class_<PyMCTSNode>(m, "MCTSNode")
        .def_property_readonly("parent", &PyMCTSNode::parent)
        .def_property_readonly("children", &PyMCTSNode::children)
        .def_property_readonly("fen", &PyMCTSNode::fen)
        .def_property_readonly("move", &PyMCTSNode::move)
        .def_property_readonly("visits", &PyMCTSNode::visits)
        .def_property_readonly("virtual_loss", &PyMCTSNode::virtual_loss)
        .def_property_readonly("result", &PyMCTSNode::result)
        .def_property_readonly("policy", &PyMCTSNode::policy)
        .def("ucb", &PyMCTSNode::ucb, py::arg("cParam"), py::arg("parentNumberOfVisits"),
             R"pbdoc(
                 Calculate the UCB score for this node.
                 `cParam` is the exploration parameter, and `parentNumberOfVisits` is the number of visits
                 to the parent node.
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
