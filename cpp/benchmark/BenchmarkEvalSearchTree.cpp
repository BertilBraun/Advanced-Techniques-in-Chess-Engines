#include "position.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#ifdef __linux__
#include <unistd.h>
#endif

#include "BoardEncoding.hpp"
#include "MCTS/EvalMCTSNode.h"
#include "MCTS/EvalSearchTree.hpp"

namespace {
struct BenchmarkResult {
    std::string variant;
    std::uint64_t searches;
    double elapsedSeconds;
    std::size_t liveNodes;
    std::size_t liveEdges;
    std::size_t residentBytes;
};

void initializeStockfish() {
    Bitboards::init();
    Position::init();
}

std::vector<MoveScore> uniformPolicy(const Board &board) {
    const std::vector<Move> &legalMoves = board.validMoves();
    std::vector<MoveScore> moves;
    moves.reserve(legalMoves.size());
    const float policy = 1.0F / static_cast<float>(legalMoves.size());
    for (const Move move : legalMoves) {
        moves.emplace_back(move, policy);
    }
    return moves;
}

float deterministicValue(const Board &board) {
    const int bucket = static_cast<int>(board.validMoves().size() % 21U) - 10;
    return static_cast<float>(bucket) / 10.0F;
}

std::size_t residentBytes() {
#ifdef __linux__
    std::ifstream statistics("/proc/self/statm");
    std::size_t totalPages = 0;
    std::size_t residentPages = 0;
    statistics >> totalPages >> residentPages;
    return residentPages * static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
#else
    return 0;
#endif
}

std::pair<std::size_t, std::size_t> pointerTreeSize(const std::shared_ptr<EvalMCTSNode> &root) {
    std::size_t nodes = 0;
    std::size_t edges = 0;
    std::vector<std::shared_ptr<EvalMCTSNode>> pending{root};
    while (!pending.empty()) {
        const std::shared_ptr<EvalMCTSNode> current = pending.back();
        pending.pop_back();
        ++nodes;
        const EvalMCTSNode::ChildSnapshot children = current->children();
        if (children != nullptr) {
            edges += children->size();
            pending.insert(pending.end(), children->begin(), children->end());
        }
    }
    return {nodes, edges};
}

BenchmarkResult benchmarkPointerTree(const std::uint64_t searches) {
    const std::shared_ptr<EvalMCTSNode> root = EvalMCTSNode::createRoot(Board{});
    const auto startedAt = std::chrono::steady_clock::now();
    for (std::uint64_t search = 0; search < searches; ++search) {
        std::shared_ptr<EvalMCTSNode> leaf = root;
        while (leaf->isExpanded()) {
            leaf = leaf->bestChild(2.0F);
        }
        leaf->materializeBoard();
        if (leaf->isTerminal()) {
            leaf->backPropagate(getBoardResultScore(leaf->board()));
        } else {
            leaf->expand(uniformPolicy(leaf->board()));
            leaf->backPropagate(deterministicValue(leaf->board()));
        }
    }
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - startedAt).count();
    const auto [nodes, edges] = pointerTreeSize(root);
    return {"pointer", searches, elapsed, nodes, edges, residentBytes()};
}

BenchmarkResult benchmarkArena(const std::uint64_t searches, const EvalEdgeStorage storage,
                               std::string variant) {
    EvalSearchTree tree(Board{}, 1'024, storage);
    const auto startedAt = std::chrono::steady_clock::now();
    for (std::uint64_t search = 0; search < searches; ++search) {
        const EvalNodeIndex leaf = tree.selectLeaf(2.0F);
        if (tree.node(leaf).isTerminal()) {
            tree.backPropagate(leaf, getBoardResultScore(tree.node(leaf).board));
        } else {
            tree.expand(leaf, uniformPolicy(tree.node(leaf).board));
            tree.backPropagate(leaf, deterministicValue(tree.node(leaf).board));
        }
    }
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - startedAt).count();
    return {std::move(variant),    searches,       elapsed, tree.liveNodeCount(),
            tree.totalEdgeCount(), residentBytes()};
}

void printResult(const BenchmarkResult &result) {
    std::cout << "variant=" << result.variant << " searches=" << result.searches
              << " elapsed_seconds=" << result.elapsedSeconds << " searches_per_second="
              << static_cast<double>(result.searches) / result.elapsedSeconds
              << " live_nodes=" << result.liveNodes << " live_edges=" << result.liveEdges
              << " resident_bytes=" << result.residentBytes << '\n';
}

void printProvenance() {
    std::cout << "compiler=" << __VERSION__ << " cxx_standard=" << __cplusplus
              << " hardware_threads=" << std::thread::hardware_concurrency()
              << " sizeof_eval_edge=" << sizeof(EvalSearchEdge)
              << " sizeof_eval_node=" << sizeof(EvalSearchNode)
              << " sizeof_pointer_node=" << sizeof(EvalMCTSNode) << '\n';
}
} // namespace

int main(const int argumentCount, const char *const arguments[]) {
    initializeStockfish();
    const std::string variant = argumentCount >= 2 ? arguments[1] : "all";
    const std::uint64_t searches = argumentCount >= 3 ? std::stoull(arguments[2]) : 100'000ULL;
    if (searches == 0) {
        throw std::invalid_argument("search count must be positive");
    }

    printProvenance();
    if (variant == "pointer" || variant == "all") {
        printResult(benchmarkPointerTree(searches));
    }
    if (variant == "arena-new-delete" || variant == "all") {
        printResult(benchmarkArena(searches, EvalEdgeStorage::NewDelete, "arena-new-delete"));
    }
    if (variant == "arena-pool" || variant == "all") {
        printResult(benchmarkArena(searches, EvalEdgeStorage::UnsynchronizedPool, "arena-pool"));
    }
    if (variant != "all" && variant != "pointer" && variant != "arena-new-delete" &&
        variant != "arena-pool") {
        throw std::invalid_argument(
            "variant must be pointer, arena-new-delete, arena-pool, or all");
    }
    return 0;
}
