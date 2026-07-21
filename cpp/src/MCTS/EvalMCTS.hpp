#pragma once

#include "common.hpp"

#include "EvalMCTSNode.h"

#include "../InferenceClient.hpp"
#include "util/ThreadPool.h"

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
        : m_client(clientArgs), m_args(mctsArgs), m_threadPool(mctsArgs.num_threads) {}

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
    ThreadPool m_threadPool;

    [[nodiscard]] EvalMCTSResult
    search(const std::shared_ptr<EvalMCTSNode> &root,
           const std::optional<std::chrono::steady_clock::time_point> deadline,
           const std::optional<int> searchLimit) {
        std::atomic<int> claimed{0};
        std::atomic<int> completed{0};

        auto worker = [&] {
            thread_local std::vector<const Board *> oneBoard{nullptr}; // size = 1

            while (true) {
                if (deadline.has_value() && std::chrono::steady_clock::now() >= *deadline) {
                    break;
                }
                int claim = claimed.load(std::memory_order_relaxed);
                while (true) {
                    if (searchLimit.has_value() && claim >= *searchLimit) {
                        return;
                    }
                    if (claimed.compare_exchange_weak(claim, claim + 1,
                                                      std::memory_order_relaxed)) {
                        break;
                    }
                }
                /* 1) selection */
                const auto leaf = getBestChildOrBackPropagate(root);
                if (!leaf) {
                    completed.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }

                leaf->addVirtualLoss();
                oneBoard[0] = &leaf->board; // reuse small buffer

                /* 2) immediate single-board inference
                       (merged into a large batch inside InferenceClient) */
                try {
                    const std::vector<InferenceResult> inferenceResults =
                        m_client.inferenceBatch(oneBoard);
                    const InferenceResult &inferenceResult = inferenceResults.front();

                    /* 3) expansion + backup */
                    leaf->expand(inferenceResult.moves, inferenceResult.outcome);
                    leaf->backPropagateAndRemoveVirtualLoss(inferenceResult.value());
                } catch (...) {
                    leaf->cancelVirtualLoss();
                    throw;
                }
                completed.fetch_add(1, std::memory_order_relaxed);
            }
        };

        const size_t P = std::max<size_t>(1, m_threadPool.numThreads());
        std::vector<std::future<void>> futures;
        futures.reserve(P);
        for (int _ : range(P))
            futures.emplace_back(m_threadPool.enqueue(worker));

        std::exception_ptr workerException;
        for (auto &future : futures) {
            try {
                future.get();
            } catch (...) {
                if (workerException == nullptr) {
                    workerException = std::current_exception();
                }
            }
        }
        if (workerException != nullptr) {
            std::rethrow_exception(workerException);
        }

        const int visits = root->number_of_visits.load(std::memory_order_relaxed);
        const float result = visits == 0 ? 0.0f
                                         : root->result_sum.load(std::memory_order_relaxed) /
                                               static_cast<float>(visits);
        return EvalMCTSResult{result, root->gatherVisitCounts(), root,
                              completed.load(std::memory_order_relaxed)};
    }

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
