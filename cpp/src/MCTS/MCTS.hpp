#pragma once

#include "common.hpp"

#include "MCTSNode.hpp"

#include "../InferenceClient.hpp"
#include "util/ThreadPool.h"

typedef std::pair<int, int> VisitCount; // Mapping from Encoded Move to visit count.
typedef std::vector<VisitCount> VisitCounts;

// Contains the arguments for the MCTS algorithm.
struct MCTSParams {
    // Number of parallel strands to run the MCTS algorithm.
    // Higher values enable more parallelism but also increase exploration.
    int num_parallel_searches;

    uint32 num_full_searches; // Number of searches to perform if full searches are requested.

    uint32 num_fast_searches; // Number of searches to perform if fast searches are requested.

    // The c parameter used in the UCB1 formula to balance exploration and exploitation.
    float c_param;

    // Alpha value for the Dirichlet noise. Typically around 10/number_of_actions.
    float dirichlet_alpha;

    // Epsilon value for the Dirichlet noise added to the root node to encourage exploration.
    float dirichlet_epsilon;

    uint8 min_visit_count; // Minimum visit count for a child of a root node.

    uint8 num_threads; // Number of threads to use for parallel processing.

    MCTSParams(int num_parallel_searches, uint32 num_full_searches, uint32 num_fast_searches,
               float c_param, float dirichlet_alpha, float dirichlet_epsilon, uint8 min_visit_count,
               uint8 num_threads)
        : num_parallel_searches(num_parallel_searches), num_full_searches(num_full_searches),
          num_fast_searches(num_fast_searches), c_param(c_param), dirichlet_alpha(dirichlet_alpha),
          dirichlet_epsilon(dirichlet_epsilon), min_visit_count(min_visit_count),
          num_threads(num_threads) {}
};

struct MCTSResult {
    float result;
    VisitCounts visits;
    std::shared_ptr<MCTSNode> root;
};

struct MCTSStatistics {
    float averageDepth = 0.0f;        // Average depth of the search trees.
    float averageEntropy = 0.0f;      // Average entropy of the visit counts.
    float averageKLDivergence = 0.0f; // Average KL divergence of the visit counts.
};

struct MCTSResults {
    std::vector<MCTSResult> results;
    MCTSStatistics mctsStats;
};

/*
 * How to use:
 * 1. Create an instance of MCTS with the desired parameters on the Python side.
 * 2. Keep the instance alive while the parameters are not changed. I.e. while no new Model needs to
 * be loaded.
 * 3. While the instance is alive, you can call the `search` method with a list of boards.
 *     - Each board is represented as a tuple of (FEN string, NodeId of the child node which was
 *       previously searched (INVALID_NODE initially), number of searches that this board should be
 *       searched for in total).
 *     - Returned is a list of MCTSResult, each containing:
 *         - `result`: The average result score of the root node.
 *         - `visits`: A list of (move, visit count) pairs for each child of the root node.
 *         - `children`: A list of NodeIds for the children of the root node.
 *       - `mctsStats`: Statistics about the MCTS search, including average depth, entropy, and KL
 *                      divergence. These should be logged to TensorBoard.
 *     - The Python side should select one of the moves based on the visit counts.
 *     - The next search can then be performed on the selected move, which will reuse the Search
 *       Tree, if the child node id is provided
 * 4. If the MCTS parameters change:
 *   - First get the InferenceStatistics and log them to TensorBoard.
 *   - Delete the current MCTS instance.
 *   - Create a new instance of MCTS with the new parameters.
 *   - Search again but without reusing the previous Search Tree.
 *
 */

using BoardTuple =
    std::tuple<std::shared_ptr<MCTSNode> /* root node */, bool /*should run full searches*/>;

class MCTS {
public:
    MCTS(const InferenceClientParams &clientArgs, const MCTSParams &mctsArgs)
        : m_client(clientArgs), m_args(mctsArgs), m_threadPool(mctsArgs.num_threads) {}

    [[nodiscard]] MCTSResults search(const std::vector<BoardTuple> &boards);

    [[nodiscard]] std::pair<InferenceStatistics, TimeInfo> getInferenceStatistics();

    [[nodiscard]] InferenceClient *getInferenceClient() { return &m_client; }

    [[nodiscard]] MCTSResult evalSearch(const std::shared_ptr<MCTSNode> &root,
                                        int numberOfSearches);

private:
    InferenceClient m_client;
    MCTSParams m_args;
    ThreadPool m_threadPool;

    [[nodiscard]] std::vector<std::pair<MCTSResult, MCTSStatistics>>
    searchGames(const std::vector<BoardTuple> &boards);

    // This method performs several iterations of tree search in parallel.
    void parallelIterate(const std::vector<std::shared_ptr<MCTSNode>> &roots);

    // Add Dirichlet noise to a vector of MoveScore.
    void addNoise(const std::shared_ptr<MCTSNode> &root) const;

    [[nodiscard]] std::shared_ptr<MCTSNode>
    getBestChildOrBackPropagate(std::shared_ptr<MCTSNode> root, float cParam) const;
};
