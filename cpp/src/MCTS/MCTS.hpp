#pragma once

#include "../common.hpp"

#include "../BoardEncoding.hpp"
#include "../MoveEncoding.hpp"
#include "MCTSNode.hpp"

typedef std::vector<std::pair<std::vector<MoveScore>, float>> InferenceResult;

struct VisitCounts {
    struct VisitCount {
        int move;
        int count;
    };

    std::vector<VisitCount> visits;

    torch::Tensor actionProbabilities() {
        auto actionProbabilities = torch::zeros({ACTION_SIZE}, torch::kFloat16);
        for (const auto &visit : visits) {
            actionProbabilities[visit.move] = visit.count;
        }

        return actionProbabilities / actionProbabilities.sum();
    }
};

struct MCTSResult {
    float result;
    VisitCounts visits;
};

struct MCTSParams {
    int num_searches_per_turn;
    int num_parallel_searches;
    float c_param;
    float dirichlet_alpha;
    float dirichlet_epsilon;
};

class InferenceClient {
public:
    // The inference_batch function takes a vector of boards and returns for each board a pair:
    //   - A vector of MoveScore (the move probabilities)
    //   - A float value (the board evaluation)
    InferenceResult inference_batch(std::vector<Board> &boards) const {
        InferenceResult results;
        // TODO: Dummy implementation: for each board, return 5 moves with random policies and a
        // dummy value.
        for (auto &board : boards) {
            auto legalMoves = board.legalMoves();
            std::vector<MoveScore> moves;
            for (auto move : legalMoves) {
                moves.emplace_back(encodeMove(move),
                                   1.0f / legalMoves.size()); // Dummy uniform policy
            }
            results.emplace_back(moves, 0.0f);
        }
        return results;
    }
};

template <typename T> T lerp(T a, T b, float t) { return (1 - t) * a + t * b; }

std::vector<float> dirichlet(float alpha, size_t n) {
    // Sample from a Dirichlet distribution with parameter alpha.
    std::mt19937 random_engine(std::random_device{}());
    std::gamma_distribution<float> gamma(alpha, 1.0);

    std::vector<float> noise(n);
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        noise[i] = gamma(random_engine);
        sum += noise[i];
    }

    float norm = 1.0f / sum;
    for (size_t i = 0; i < n; i++) {
        noise[i] *= norm;
    }

    return noise;
}

class MCTS {
public:
    MCTS(InferenceClient *client, const MCTSParams &args) : client(client), args(args) {}

    std::vector<MCTSResult> search(std::vector<Board> &boards) const {
        if (boards.empty())
            return {};

        // Get policy moves (with noise) for the given boards.
        std::vector<std::vector<MoveScore>> movesList = _get_policy_with_noise(boards);

        std::vector<MCTSNode> roots;
        for (const auto &[board, moves] : zip(boards, movesList)) {
            // Since no node is pre-expanded, create a new root.
            MCTSNode root = MCTSNode::root(board);
            root.expand(moves);
            roots.push_back(std::move(root));
        }

        int iterations = args.num_searches_per_turn / args.num_parallel_searches;
        for (int i = 0; i < iterations; i++) {
            parallel_iterate(roots);
        }

        // Build and return the results.
        std::vector<MCTSResult> results;
        for (const MCTSNode &root : roots) {
            VisitCounts visitCounts;
            for (const MCTSNode &child : root.children) {
                visitCounts.visits.emplace_back(child.encoded_move_to_get_here,
                                                child.number_of_visits);
            }
            results.push_back({root.result_score, visitCounts});
        }

        return results;
    }

    // This method performs several iterations of tree search in parallel.
    void parallel_iterate(std::vector<MCTSNode> &roots) const {
        std::vector<MCTSNode *> nodes;
        for (int i = 0; i < args.num_parallel_searches; i++) {
            for (MCTSNode &root : roots) {
                std::optional<MCTSNode *> node =
                    _get_best_child_or_back_propagate(root, args.c_param);
                if (node.has_value()) {
                    node.value()->updateVirtualLoss(1);
                    nodes.push_back(node.value());
                }
            }
        }
        if (nodes.empty())
            return;

        // Gather boards for inference.
        std::vector<Board> boards;
        boards.reserve(nodes.size());
        for (MCTSNode *node : nodes)
            boards.push_back(node->board);

        // Run inference in batch.
        InferenceResult results = client->inference_batch(boards);

        for (auto [node, result] : zip(nodes, results)) {
            const auto &[moves, value] = result;
            node->expand(moves);
            node->backPropagate(value);
            node->updateVirtualLoss(-1);
        }
    }

private:
    InferenceClient *client;
    MCTSParams args;

    // Get policy moves with added Dirichlet noise.
    std::vector<std::vector<MoveScore>> _get_policy_with_noise(std::vector<Board> &boards) const {
        InferenceResult inferenceResults = client->inference_batch(boards);

        std::vector<std::vector<MoveScore>> noisyMoves;
        for (auto &result : inferenceResults) {
            noisyMoves.push_back(_add_noise(result.first));
        }
        return noisyMoves;
    }

    // Add Dirichlet noise to a vector of MoveScore.
    std::vector<MoveScore> _add_noise(const std::vector<MoveScore> &moves) const {
        std::vector<float> noiseList = dirichlet(args.dirichlet_alpha, moves.size());

        std::vector<MoveScore> noisyMoves;
        noisyMoves.reserve(moves.size());
        for (const auto &[noise, moveScore] : zip(noiseList, moves)) {
            const auto &[move, score] = moveScore;
            float newScore = lerp(score, noise, args.dirichlet_epsilon);
            noisyMoves.push_back({move, newScore});
        }

        return noisyMoves;
    }

    // Traverse the tree to find the best child or, if the node is terminal,
    // back-propagate the board’s result.
    std::optional<MCTSNode *> _get_best_child_or_back_propagate(MCTSNode &root,
                                                                float c_param) const {
        MCTSNode *node = &root;
        while (node->isFullyExpanded()) {
            node = &node->bestChild(c_param);
        }

        if (node->isTerminalNode()) {
            auto result = getBoardResultScore(node->board);
            assert(result.has_value());
            node->backPropagate(result.value());
            return std::nullopt;
        }
        return {node};
    }
};

/*
void log_details(const std::vector<MCTSNode> &roots) {

    // Compute the depth of each search tree using a DFS.
    auto dfs = [&](MCTSNode *node, auto &dfs_ref) -> int {
        if (!node || !node->is_fully_expanded())
            return 0;
        int max_depth = 0;
        for (MCTSNode *child : node->children) {
            int d = dfs_ref(child, dfs_ref);
            if (d > max_depth)
                max_depth = d;
        }
        return 1 + max_depth;
    };
    std::vector<int> depths;
    for (MCTSNode *root : roots) {
        depths.push_back(dfs(root, dfs));
    }
    float average_depth = sum(depths) / depths.size();
    log_scalar("dataset/average_search_depth", average_depth);

    // Compute the entropy of the visit counts.
    auto entropy = [](MCTSNode *node) -> float {
        int total = node->number_of_visits();
        float ent = 0.0f;
        for (int v : node->children_number_of_visits) {
            if (v > 0) {
                float p = static_cast<float>(v) / total;
                ent -= p * std::log2(p);
            }
        }
        return ent;
    };
    std::vector<float> entropies;
    for (MCTSNode *root : roots) {
        entropies.push_back(entropy(root));
    }
    float average_entropy = sum(entropies) / entropies.size();
    log_scalar("dataset/average_search_entropy", average_entropy);
}
    */