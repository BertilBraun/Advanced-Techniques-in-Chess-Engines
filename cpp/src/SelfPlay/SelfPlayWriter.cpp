#include "SelfPlayWriter.hpp"

CompressedEncodedBoard _flipBoardVertical(const CompressedEncodedBoard &board) {
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

VisitCounts _flipActionProbabilitiesVertical(const VisitCounts &visitCounts) {
    VisitCounts flippedVisitCounts;
    for (const auto &[move, count] : visitCounts.visits) {
        const Move flippedMove = decodeMove(move);
        const auto [moveFromRow, moveFromCol] = squareToIndex(flippedMove.fromSquare());
        const auto [moveToRow, moveToCol] = squareToIndex(flippedMove.toSquare());
        const Square flippedMoveFrom = square(moveFromRow, BOARD_LENGTH - 1 - moveFromCol);
        const Square flippedMoveTo = square(moveToRow, BOARD_LENGTH - 1 - moveToCol);
        const int flippedIndex =
            encodeMove(Move(flippedMoveFrom, flippedMoveTo, flippedMove.promotion()));
        flippedVisitCounts.visits.push_back({flippedIndex, count});
    }
    return flippedVisitCounts;
}

std::vector<std::pair<CompressedEncodedBoard, VisitCounts>>
_symmetricVariations(const CompressedEncodedBoard &board, const VisitCounts &visitCounts) {
    std::vector<std::pair<CompressedEncodedBoard, VisitCounts>> variations;

    auto flippedBoard = _flipBoardVertical(board);
    auto flippedVisitCounts = _flipActionProbabilitiesVertical(visitCounts);

    variations.push_back({board, visitCounts});
    variations.push_back({flippedBoard, flippedVisitCounts});

    return variations;
}

VisitCounts _preprocessVisitCounts(const VisitCounts &visitCounts) {
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

void SelfPlayWriter::write(const SelfPlayGame &game, float outcome, bool resignation,
                           bool tooLong) {
    std::lock_guard<std::mutex> lock(m_mutex);

    // Update per-game statistics.
    m_stats.num_games += 1;
    m_stats.game_lengths += game.memory.size();
    m_stats.total_generation_time += game.generationTime();

    if (resignation)
        m_stats.resignations += 1;
    if (tooLong)
        m_stats.num_too_long_games += 1;

    if (rand() % 100 == 0)
        _logGame(game, outcome);

    // Process moves in reverse order.
    for (auto mem : reverse(game.memory)) {
        const CompressedEncodedBoard encodedBoard = encodeBoard(mem.board);
        // Adjust outcome based on turn.
        const float turnGameOutcome = (game.board.turn == mem.board.turn) ? outcome : -outcome;
        const float resultScore =
            clamp(turnGameOutcome + m_args.self_play.result_score_weight * mem.result, -1.0f, 1.0f);

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
void SelfPlayWriter::updateIteration(int iteration) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (!m_samples.empty()) {
        // Flush any remaining samples to file.
        // This is important to ensure that all samples are written before the next iteration.
        _flushBatch();
    }

    // Log statistics to TensorBoard.
    if (m_stats.num_samples > 0) {
        m_logger.add_scalar("dataset/num_samples", iteration, (float) m_stats.num_samples);
        m_logger.add_scalar("dataset/num_games", iteration, (float) m_stats.num_games);
        m_logger.add_scalar("dataset/average_game_lengths", iteration,
                            (float) m_stats.game_lengths / m_stats.num_games);
        m_logger.add_scalar("dataset/average_generation_time", iteration,
                            m_stats.total_generation_time / m_stats.num_games);
        m_logger.add_scalar("dataset/resignations", iteration, (float) m_stats.resignations);
        m_logger.add_scalar("dataset/num_too_long_games", iteration,
                            (float) m_stats.num_too_long_games);
        m_logger.add_scalar("dataset/num_samples_per_game", iteration,
                            (float) m_stats.num_samples / m_stats.num_games);

        log("Iteration", iteration, ":");
        log("  num_samples:", m_stats.num_samples);
        log("  num_games:", m_stats.num_games);
        log("  average_game_lengths:", (float) m_stats.game_lengths / m_stats.num_games);
        log("  average_generation_time:", m_stats.total_generation_time / m_stats.num_games);
        log("  resignations:", m_stats.resignations);
        log("  num_too_long_games:", m_stats.num_too_long_games);
        log("  num_samples_per_game:", (float) m_stats.num_samples / m_stats.num_games);
    }

    // Update the iteration for the current iteration.
    m_iteration = iteration;
    // The next batch will start from 0.
    m_batchCounter = 0;

    // resets all statistics to defaults
    m_stats = Stats();
}
void SelfPlayWriter::_addSample(const CompressedEncodedBoard &board, const VisitCounts &visitCounts,
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
std::string SelfPlayWriter::_getSaveFilename() {
    assert(m_iteration != -1);
    const std::string saveFolder = m_args.save_path + "/iteration_" + std::to_string(m_iteration);

    // Ensure the save path exists.
    if (!std::filesystem::exists(saveFolder)) {
        std::filesystem::create_directories(saveFolder);
    }

    std::string filename =
        saveFolder + "/" + m_args.writer.filePrefix + "_" + std::to_string(m_batchCounter) + ".bin";
    m_batchCounter += 1;

    // while file already exists, update filename and increase batchCounter
    while (std::filesystem::exists(std::filesystem::path{filename})) {
        filename = saveFolder + "/" + m_args.writer.filePrefix + "_" +
                   std::to_string(m_batchCounter) + ".bin";
        m_batchCounter += 1;
    }

    return filename;
}
void SelfPlayWriter::_flushBatch() {
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
void SelfPlayWriter::_logGame(const SelfPlayGame &game, float result) {
    auto step = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    // Log the game moves and result.
    std::string moves;
    for (const auto &move : game.playedMoves) {
        moves += std::to_string(encodeMove(move)) + ",";
    }
    moves.pop_back(); // Remove the trailing comma
    m_logger.add_text("moves/" + std::to_string(m_iteration), step,
                      (std::to_string(result) + ':' + moves).c_str());
}
