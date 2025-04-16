#include "SelfPlayWriter.hpp"
#include "BoardEncoding.hpp"

const std::string BOARD_FILE_POSTFIX = "_board.csv";
const std::string MOVE_FILE_POSTFIX = "_moves.csv";
const std::string STATS_FILE_POSTFIX = "_stats.json";

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
    if (visitCounts.empty()) {
        return {};
    }

    VisitCounts flippedVisitCounts;
    flippedVisitCounts.reserve(visitCounts.size());
    for (const auto &[move, count] : visitCounts) {
        const Move flippedMove = decodeMove(move);
        const auto [moveFromRow, moveFromCol] = squareToIndex(flippedMove.fromSquare());
        const auto [moveToRow, moveToCol] = squareToIndex(flippedMove.toSquare());
        const Square flippedMoveFrom = square(moveFromRow, BOARD_LENGTH - 1 - moveFromCol);
        const Square flippedMoveTo = square(moveToRow, BOARD_LENGTH - 1 - moveToCol);
        const int flippedIndex =
            encodeMove(Move(flippedMoveFrom, flippedMoveTo, flippedMove.promotion()));
        flippedVisitCounts.emplace_back(flippedIndex, count);
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
    if (visitCounts.empty()) {
        return {};
    }

    int totalVisits = 0;
    for (const auto &[_, visitCount] : visitCounts) {
        totalVisits += visitCount;
    }

    VisitCounts newVisitCounts;
    newVisitCounts.reserve(visitCounts.size());
    // Filter out moves with less than 1% of the total visits.
    // This is to reduce the size of the visit counts and focus on the most significant moves.
    for (const auto &[move, count] : visitCounts) {
        if (count >= totalVisits * 0.01) {
            newVisitCounts.emplace_back(move, count);
        }
    }

    return newVisitCounts;
}

void SelfPlayWriter::write(const SelfPlayGame &game, float outcome, bool resignation,
                           bool tooLong) {
    std::lock_guard<std::mutex> lock(m_mutex);
    log("Writing game with outcome:", outcome, "Number of moves:", game.memory.size(),
        "resignation:", resignation, "tooLong:", tooLong, "generationTime:", game.generationTime(),
        "playedMoves:", game.playedMoves.size(), "iteration:", m_iteration);
    log(game.playedMoves);
    reset_times(nullptr, m_iteration);

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

    if (m_samples.size() >= m_args.writer.batchSize) {
        _flushBatch();
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
}

std::string SelfPlayWriter::_getSaveFilename() {
    assert(m_iteration != -1);
    const std::string saveFolder = m_args.save_path + "/iteration_" + std::to_string(m_iteration);

    // Ensure the save path exists.
    if (!std::filesystem::exists(saveFolder)) {
        std::filesystem::create_directories(saveFolder);
    }

    std::string filename = saveFolder + "/" + m_args.writer.filePrefix + "_" +
                           std::to_string(m_batchCounter) + STATS_FILE_POSTFIX;
    m_batchCounter += 1;

    // while file already exists, update filename and increase batchCounter
    while (std::filesystem::exists(std::filesystem::path{filename})) {
        filename = saveFolder + "/" + m_args.writer.filePrefix + "_" +
                   std::to_string(m_batchCounter) + STATS_FILE_POSTFIX;
        m_batchCounter += 1;
    }

    // Remove the postfix from the filename.
    return filename.substr(0, filename.size() - STATS_FILE_POSTFIX.size());
}

void SelfPlayWriter::_flushBatch() {
    std::string baseFilename = _getSaveFilename();

    // === Write Stats JSON to {baseFilename}_stats.json ===
    {
        std::string statsFilename = baseFilename + STATS_FILE_POSTFIX;
        std::ofstream statsOut(statsFilename);
        if (!statsOut) {
            log("Failed to open file for writing:", statsFilename);
            return;
        }

        nlohmann::json j;
        j["num_samples"] = m_stats.num_samples;
        j["num_games"] = m_stats.num_games;
        j["game_lengths"] = m_stats.game_lengths;
        j["total_generation_time"] = m_stats.total_generation_time;
        j["resignations"] = m_stats.resignations;
        j["num_too_long_games"] = m_stats.num_too_long_games;

        // Write pretty JSON with an indent of 4 spaces.
        statsOut << j.dump(4);
        statsOut.close();
    }

    // === Write Boards as binary to {baseFilename}_boards.csv ===
    {
        std::string boardsFilename = baseFilename + BOARD_FILE_POSTFIX;
        std::ofstream boardsOut(boardsFilename);
        if (!boardsOut) {
            log("Failed to open file for writing:", boardsFilename);
            return;
        }

        for (const auto &sample : m_samples) {
            // Write the board as a binary string.
            boardsOut << std::hex << std::setfill('0');
            for (int i : range(ENCODING_CHANNELS)) {
                boardsOut << std::setw(16) << sample.board[i] << ",";
            }
            boardsOut << sample.resultScore << "\n";
        }
        boardsOut.close();
    }

    // === Write Visit Counts (Moves) to {baseFilename}_moves.csv ===
    {
        std::string movesFilename = baseFilename + MOVE_FILE_POSTFIX;
        std::ofstream movesOut(movesFilename);
        if (!movesOut) {
            std::cerr << "Failed to open file for writing: " << movesFilename << std::endl;
            return;
        }

        // Optionally write a CSV header:
        movesOut << "sample_index,move,count\n";

        // Write each sample’s visit count pairs.
        for (const auto &[sampleIndex, sample] : enumerate(m_samples)) {
            for (const auto &[move, count] : sample.visitCounts) {
                // Optionally, you could perform your validations here.
                movesOut << sampleIndex << "," << move << "," << count << "\n";
            }
        }
        movesOut.close();
    }

    // Clear the batch and reset per-batch statistics.
    m_samples.clear();
    m_stats = Stats();
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
