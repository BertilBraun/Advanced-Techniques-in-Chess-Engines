#pragma once

#include <functional>
#include <optional>
#include <string>

// Contains the arguments for the MCTS algorithm.
struct MCTSParams {
    // The maximum number of searches to run the MCTS algorithm in self-play.
    // At each move, the algorithm is run this many times to decide the next move.
    int num_searches_per_turn;

    // Number of parallel strands to run the MCTS algorithm.
    // Higher values enable more parallelism but also increase exploration.
    int num_parallel_searches;

    // The c parameter used in the UCB1 formula to balance exploration and exploitation.
    double c_param;

    // Alpha value for the Dirichlet noise. Typically around 10/number_of_actions.
    double dirichlet_alpha;

    // Epsilon value for the Dirichlet noise added to the root node to encourage exploration.
    double dirichlet_epsilon;
};

// Contains the parameters for self-play.
struct SelfPlayParams {
    MCTSParams mcts;

    // The number of games to run in parallel for self-play.
    int num_parallel_games;

    // After this many moves, the search will play greedily rather than using temperature sampling.
    int num_moves_after_which_to_play_greedy;

    // The maximum number of moves in a game before it is considered a draw.
    int max_moves;

    // Sampling temperature for move selection during self-play.
    // A temperature of 1.0 is the same as the raw policy; 0.0 is pure argmax.
    double temperature = 1.25;

    // Weight for interpolating between the final game outcome and the MCTS result score.
    double result_score_weight = 0.5;

    // Number of games to collect before writing them to disk.
    int num_games_after_which_to_write = 5;

    // Resignation threshold; if the MCTS result score falls below this, the game is resigned.
    double resignation_threshold = -0.85;
};

// Contains the top-level training arguments.
struct TrainingArgs {
    // Path to save the model, training logs, etc. after each iteration.
    std::string save_path;

    SelfPlayParams self_play;
};
