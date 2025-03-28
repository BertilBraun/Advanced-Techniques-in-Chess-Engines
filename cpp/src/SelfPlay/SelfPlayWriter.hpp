#pragma once

#include "BoardEncoding.hpp"
#include "MCTS/VisitCounts.hpp"
#include "MoveEncoding.hpp"
#include "SelfPlayGame.hpp"
#include "common.hpp"
#include "util/json.hpp"
#include <filesystem>
#include <mutex>

class SelfPlayWriter {
public:
    // Constructor takes a file prefix, the batch size, and optional user metadata.
    SelfPlayWriter(SelfPlayParams selfPlayArgs) : m_args(selfPlayArgs), m_batchCounter(0) {}

    ~SelfPlayWriter() {
        if (!m_samples.empty()) {
            _flushBatch();
        }
    }

    // Process one game and update the statistics.
    void write(const SelfPlayGame &game, float outcome, bool resignation, bool tooLong) {
        std::lock_guard<std::mutex> lock(m_mutex);

        // Update per-game statistics.
        m_stats.num_games += 1;
        m_stats.game_lengths += game.memory.size();
        m_stats.total_generation_time += game.generationTime();

        if (resignation) {
            m_stats.resignations += 1;
        }
        if (tooLong) {
            m_stats.num_too_long_games += 1;
        }

        // Process moves in reverse order.
        for (auto mem : reverse(game.memory)) {
            const CompressedEncodedBoard encodedBoard = encodeBoard(mem.board);
            // Adjust outcome based on turn.
            const float turnGameOutcome = (game.board.turn == mem.board.turn) ? outcome : -outcome;
            const float resultScore =
                clamp(turnGameOutcome + m_args.result_score_weight * mem.result, -1.0f, 1.0f);

            // Process symmetric variations.
            auto variations =
                _symmetricVariations(encodedBoard, _preprocessVisitCounts(mem.visitCounts));
            for (const auto &[board, visitCounts] : variations) {
                _addSample(board, visitCounts, resultScore);
                m_stats.num_samples += 1;
            }
            outcome *= 0.997f; // discount the game outcome for each move
        }
    }

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
                    float resultScore) {
        Sample sample;
        sample.board = board;
        sample.visitCounts = visitCounts;
        sample.resultScore = resultScore;
        m_samples.push_back(std::move(sample));

        if (m_samples.size() >= m_args.writer.batchSize) {
            _flushBatch();
        }
    }

    std::string _getSaveFilename() {
        std::string filename =
            m_args.writer.filePrefix + "_" + std::to_string(m_batchCounter) + ".bin";
        m_batchCounter += 1;

        // while file already exists, update filename and increase batchCounter
        while (std::filesystem::exists(std::filesystem::path{filename})) {
            filename = m_args.writer.filePrefix + "_" + std::to_string(m_batchCounter) + ".bin";
            m_batchCounter += 1;
        }

        return filename;
    }

    // Flush the current batch to file.
    void _flushBatch() {
        std::string filename = _getSaveFilename();
        std::ofstream out(filename, std::ios::binary);
        if (!out) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return;
        }

        auto add = [&](const auto &data, size_t size = 0) {
            if (size == 0) {
                size = sizeof(data);
            }
            out.write(reinterpret_cast<const char *>(&data), size);
        };

        // --- Build Metadata JSON from the Stats struct ---
        nlohmann::json j;
        j["num_samples"] = m_stats.num_samples;
        j["num_games"] = m_stats.num_games;
        j["game_lengths"] = m_stats.game_lengths;
        j["total_generation_time"] = m_stats.total_generation_time;
        j["resignations"] = m_stats.resignations;
        j["num_too_long_games"] = m_stats.num_too_long_games;
        std::string metadata_str = j.dump(4);
        uint32_t metadataLength = static_cast<uint32_t>(metadata_str.size());

        // --- Write Header ---
        // Magic number (4 bytes).
        add("SMPF", 4);
        // Version (4 bytes, here version 1).
        uint32_t version = 1;
        add(version);
        // Metadata JSON length (4 bytes).
        add(metadataLength);
        // Metadata JSON string.
        add(metadata_str.c_str(), metadataLength);

        // --- Write Sample Count ---
        uint32_t sampleCount = static_cast<uint32_t>(m_samples.size());
        add(sampleCount);

        // --- Write Each Sample ---
        for (const auto &sample : m_samples) {
            // Write the fixed board array (14 x 64-bit integers).
            add(sample.board.data(), sizeof(uint64_t) * ENCODING_CHANNELS);

            // Write the number of pairs in visitCounts (as a 32-bit integer).
            uint32_t numPairs = static_cast<uint32_t>(sample.visitCounts.visits.size());
            add(numPairs);

            // Write each pair (each int as 32-bit).
            for (const auto &[move, count] : sample.visitCounts.visits) {
                add(static_cast<int32_t>(move));
                add(static_cast<int32_t>(count));
            }

            // Write the result score as a 32-bit float.
            add(sample.resultScore);
        }

        out.close();

        // Clear the batch and reset per-batch statistics.
        m_samples.clear();
        m_stats = Stats(); // resets all statistics to defaults
        ++m_batchCounter;
    }

    CompressedEncodedBoard _flipBoardVertical(const CompressedEncodedBoard &board) const {
        EncodedBoard flippedBoard;
        EncodedBoard decompressedBoard = decompress(board);

        for (int channel : range(ENCODING_CHANNELS)) {
            for (int row : range(BOARD_LENGTH)) {
                for (int col : range(BOARD_LENGTH)) {
                    flippedBoard[channel][row][col] =
                        decompressedBoard[channel][row][BOARD_LENGTH - 1 - col];
                }
            }
        }

        return compress(flippedBoard);
    }

    VisitCounts _flipActionProbabilitiesVertical(const VisitCounts &visitCounts) const {
        VisitCounts flippedVisitCounts;
        for (const auto &[move, count] : visitCounts.visits) {
            Move flippedMove = decodeMove(move);
            auto [moveFromRow, moveFromCol] = squareToIndex(flippedMove.fromSquare());
            auto [moveToRow, moveToCol] = squareToIndex(flippedMove.toSquare());
            Square flippedMoveFrom = square(moveFromRow, BOARD_LENGTH - 1 - moveFromCol);
            Square flippedMoveTo = square(moveToRow, BOARD_LENGTH - 1 - moveToCol);
            int flippedIndex =
                encodeMove(Move(flippedMoveFrom, flippedMoveTo, flippedMove.promotion()));
            flippedVisitCounts.visits.push_back({flippedIndex, count});
        }
        return flippedVisitCounts;
    }

    std::vector<std::pair<CompressedEncodedBoard, VisitCounts>>
    _symmetricVariations(const CompressedEncodedBoard &board,
                         const VisitCounts &visitCounts) const {
        std::vector<std::pair<CompressedEncodedBoard, VisitCounts>> variations;

        auto flippedBoard = _flipBoardVertical(board);
        auto flippedVisitCounts = _flipActionProbabilitiesVertical(visitCounts);

        variations.push_back({board, visitCounts});
        variations.push_back({flippedBoard, flippedVisitCounts});

        return variations;
    }

    VisitCounts _preprocessVisitCounts(const VisitCounts &visitCounts) const {
        int totalVisits = 0;
        for (const auto &[_, visitCount] : visitCounts.visits) {
            totalVisits += visitCount;
        }

        VisitCounts newVisitCounts;
        for (const auto &[move, count] : visitCounts.visits) {
            if (count >= totalVisits * 0.01) {
                newVisitCounts.visits.push_back({move, count});
            }
        }

        return newVisitCounts;
    }

private:
    SelfPlayParams m_args;

    // Members for sample batch and file naming.
    size_t m_batchCounter;
    std::vector<Sample> m_samples;

    // Statistics for the current batch.
    Stats m_stats;

    std::mutex m_mutex; // Mutex for thread safety
};
