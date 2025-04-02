#pragma once

#include "TrainingArgs.hpp"
#include "common.hpp"

#include "BoardEncoding.hpp"
#include "MCTS/VisitCounts.hpp"
#include "MoveEncoding.hpp"
#include "SelfPlayGame.hpp"
#include "util/json.hpp"

class SelfPlayWriter {
public:
    // Constructor takes a file prefix, the batch size, and optional user metadata.
    SelfPlayWriter(TrainingArgs args, TensorBoardLogger &logger)
        : m_args(args), m_logger(logger), m_batchCounter(0) {}

    ~SelfPlayWriter() {
        if (!m_samples.empty()) {
            _flushBatch();
        }
    }

    // Process one game and update the statistics.
    void write(const SelfPlayGame &game, float outcome, bool resignation, bool tooLong);

    void updateIteration(int iteration);

private:
    // A structure to hold one sample.
    struct Sample {
        CompressedEncodedBoard board;
        VisitCounts visitCounts;
        float resultScore;
    };

    // Structure to hold metadata statistics.
    struct Stats {
        size_t num_samples = 0;
        size_t num_games = 0;
        size_t game_lengths = 0;
        double total_generation_time = 0.0;
        size_t resignations = 0;
        size_t num_too_long_games = 0;
    };

    // Add one sample to the batch.
    void _addSample(const CompressedEncodedBoard &board, const VisitCounts &visitCounts,
                    float resultScore);

    std::string _getSaveFilename();

    // Flush the current batch to file.
    void _flushBatch();

    void _logGame(const SelfPlayGame &game, float result);

    TrainingArgs m_args;
    int m_iteration = -1;
    TensorBoardLogger &m_logger;

    // Members for sample batch and file naming.
    size_t m_batchCounter;
    std::vector<Sample> m_samples;

    // Statistics for the current batch.
    Stats m_stats;

    std::mutex m_mutex; // Mutex for thread safety
};
