#pragma once

#include "common.hpp"

#include "EvalMCTSNode.h"

#include "../InferenceClient.hpp"

struct EvalMCTSParams {
    // The c parameter used in the UCB1 formula to balance exploration and exploitation.
    float c_param;

    int num_threads; // Number of threads to use for parallel processing.

    EvalMCTSParams(float c_param, int num_threads) : c_param(c_param), num_threads(num_threads) {
        if (!std::isfinite(c_param) || c_param < 0.0f) {
            throw std::invalid_argument("c_param must be finite and non-negative");
        }
        if (num_threads <= 0) {
            throw std::invalid_argument("num_threads must be positive");
        }
    }
};

struct EvalMCTSResult {
    float result;
    VisitCounts visits;
    std::shared_ptr<EvalMCTSNode> root;
    int completed_searches;
};

class EvalMCTS {
public:
    EvalMCTS(const InferenceClientParams &clientArgs, const EvalMCTSParams &mctsArgs)
        : m_client(clientArgs), m_args(mctsArgs) {}

    [[nodiscard]] InferenceResult evaluate(const Board &board) {
        const std::vector<const Board *> boards{&board};
        return m_client.inferenceBatch(boards).front();
    }

    [[nodiscard]] InferenceStatistics inferenceStatistics() { return m_client.getStatistics(); }

    [[nodiscard]] EvalMCTSResult evalSearch(const std::shared_ptr<EvalMCTSNode> &root,
                                            const int searches) {
        if (searches <= 0) {
            throw std::invalid_argument("searches must be positive");
        }
        return search(root, std::nullopt, searches);
    }

    [[nodiscard]] EvalMCTSResult
    evalSearchUntil(const std::shared_ptr<EvalMCTSNode> &root,
                    const std::chrono::steady_clock::time_point deadline,
                    const std::optional<int> searchLimit = std::nullopt) {
        if (searchLimit.has_value() && *searchLimit <= 0) {
            throw std::invalid_argument("search_limit must be positive");
        }
        return search(root, deadline, searchLimit);
    }

private:
    InferenceClient m_client;
    EvalMCTSParams m_args;

    [[nodiscard]] EvalMCTSResult
    search(const std::shared_ptr<EvalMCTSNode> &root,
           const std::optional<std::chrono::steady_clock::time_point> deadline,
           const std::optional<int> searchLimit) {
        int claimed = 0;
        int completed = 0;
        std::vector<std::shared_ptr<EvalMCTSNode>> leaves;
        std::vector<const Board *> boards;
        leaves.reserve(static_cast<size_t>(m_args.num_threads));
        boards.reserve(static_cast<size_t>(m_args.num_threads));

        while ((!deadline.has_value() || std::chrono::steady_clock::now() < *deadline) &&
               (!searchLimit.has_value() || claimed < *searchLimit)) {
            leaves.clear();
            boards.clear();
            for (int parallelSearch = 0; parallelSearch < m_args.num_threads; ++parallelSearch) {
                if ((deadline.has_value() && std::chrono::steady_clock::now() >= *deadline) ||
                    (searchLimit.has_value() && claimed >= *searchLimit)) {
                    break;
                }
                ++claimed;
                const std::shared_ptr<EvalMCTSNode> leaf = getBestChildOrBackPropagate(root);
                if (leaf == nullptr) {
                    ++completed;
                    continue;
                }
                if (std::ranges::find(leaves, leaf) != leaves.end()) {
                    --claimed;
                    break;
                }
                leaf->addVirtualLoss();
                leaves.push_back(leaf);
                boards.push_back(&leaf->board());
            }

            if (leaves.empty()) {
                continue;
            }
            size_t backedUp = 0;
            try {
                const std::vector<InferenceResult> inferenceResults =
                    m_client.inferenceBatch(boards);
                assert(inferenceResults.size() == leaves.size());
                for (size_t index = 0; index < leaves.size(); ++index) {
                    const InferenceResult &inferenceResult = inferenceResults[index];
                    leaves[index]->expand(inferenceResult.moves, inferenceResult.outcome);
                    leaves[index]->backPropagateAndRemoveVirtualLoss(inferenceResult.value());
                    ++backedUp;
                    ++completed;
                }
            } catch (...) {
                for (size_t index = backedUp; index < leaves.size(); ++index) {
                    leaves[index]->cancelVirtualLoss();
                }
                throw;
            }
        }

        const int visits = root->number_of_visits.load(std::memory_order_relaxed);
        const float result = visits == 0 ? 0.0f
                                         : root->result_sum.load(std::memory_order_relaxed) /
                                               static_cast<float>(visits);
        return EvalMCTSResult{result, root->gatherVisitCounts(), root,
                              completed};
    }

    [[nodiscard]] std::shared_ptr<EvalMCTSNode>
    getBestChildOrBackPropagate(const std::shared_ptr<EvalMCTSNode> &root) const {
        TIMEIT("MCTS::getBestChildOrBackPropagate");

        // We need to traverse the tree until we find a node that is not fully expanded
        std::shared_ptr<EvalMCTSNode> node = root;
        while (node->isExpanded())
            node = node->bestChild(m_args.c_param);

        node->materializeBoard();
        if (node->isTerminal()) {
            node->backPropagate(getBoardResultScore(node->board()));
            return nullptr;
        }
        return node;
    }
};
