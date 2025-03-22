#pragma once

#include "TrainingArgs.hpp"

static inline TrainingArgs TRAINING_ARGS = {
    .save_path = 'training_data/chess',
    .self_play =
        {
            .mcts =
                {
                    .num_searches_per_turn = 640,
                    .num_parallel_searches = 8,
                    .dirichlet_epsilon = 0.25,
                    .dirichlet_alpha = 0.3,
                    .c_param = 1.7,
                },
            .num_parallel_games = 32,
            .num_moves_after_which_to_play_greedy = 25,
            .max_moves = 250,
            .result_score_weight = 0.15,
            .num_games_after_which_to_write = 5,
            .resignation_threshold = -1.0,
        },
};