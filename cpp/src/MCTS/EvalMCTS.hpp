#pragma once

#include "common.hpp"

#include "EvalMCTSNode.h"

#include "../InferenceClient.hpp"
#include "util/ThreadPool.h"

struct EvalMCTSParams {
    // The c parameter used in the UCB1 formula to balance exploration and exploitation.
    float c_param;

    uint8 num_threads; // Number of threads to use for parallel processing.

    EvalMCTSParams(float c_param, uint8 num_threads) : c_param(c_param), num_threads(num_threads) {}
};

struct EvalMCTSResult {
    float result;
    VisitCounts visits;
    std::shared_ptr<EvalMCTSNode> root;
};

class EvalMCTS {
public:
    EvalMCTS(const InferenceClientParams &clientArgs, const EvalMCTSParams &mctsArgs)
        : m_client(clientArgs), m_args(mctsArgs), m_threadPool(mctsArgs.num_threads) {}

    [[nodiscard]] EvalMCTSResult evalSearch(const std::shared_ptr<EvalMCTSNode> &root,
                                            const int searches) {
        std::atomic<int> budget{searches};

        auto worker = [&] {
            thread_local std::vector<const Board *> oneBoard{nullptr}; // size = 1

            while (budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                /* 1) selection */
                const auto leaf = getBestChildOrBackPropagate(root);
                if (!leaf)
                    continue;

                leaf->addVirtualLoss();
                oneBoard[0] = &leaf->board; // reuse small buffer

                /* 2) immediate single-board inference
                       (merged into a large batch inside InferenceClient) */
                auto nnOut = m_client.inferenceBatch(oneBoard);
                const auto &[moves, value] = nnOut[0];

                /* 3) expansion + backup */
                leaf->expand(moves);
                leaf->backPropagateAndRemoveVirtualLoss(value);
            }
        };

        const size_t P = std::max<size_t>(1, m_threadPool.numThreads());
        std::vector<std::future<void>> futures;
        futures.reserve(P);
        for (int _ : range(P))
            futures.emplace_back(m_threadPool.enqueue(worker));

        for (auto &f : futures)
            f.get(); // wait for all threads to finish
        
        return EvalMCTSResult{root->result_sum / static_cast<float>(root->number_of_visits),
                              root->gatherVisitCounts(), root};
    }

private:
    InferenceClient m_client;
    EvalMCTSParams m_args;
    ThreadPool m_threadPool;

    [[nodiscard]] std::shared_ptr<EvalMCTSNode>
    getBestChildOrBackPropagate(const std::shared_ptr<EvalMCTSNode> &root) const {
        TIMEIT("MCTS::getBestChildOrBackPropagate");

        // We need to traverse the tree until we find a node that is not fully expanded
        std::shared_ptr<EvalMCTSNode> node = root;
        while (node->isExpanded())
            node = node->bestChild(m_args.c_param);

        if (node->isTerminal()) {
            node->backPropagate(getBoardResultScore(node->board));
            return nullptr;
        }
        return node;
    }
};
