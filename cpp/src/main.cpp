#include "common.hpp"

#include "main.hpp"

#include "MCTS/MCTS.hpp"
#include "SelfPlay/SelfPlay.hpp"
#include "SelfPlay/SelfPlayWriter.hpp"

void selfPlayMain(int runId, const std::string &savePath, int numProcessors, int numGPUs) {
    assert(runId >= 0);
    assert(numProcessors >= 1);
    assert(0 < numGPUs && numGPUs <= std::max((int) torch::cuda::device_count(), 1));
    assert(std::filesystem::exists(std::filesystem::path(savePath)));
    assert(std::filesystem::is_directory(std::filesystem::path(savePath)));

    const TrainingArgs TRAINING_ARGS = {
        .save_path = savePath,
        .self_play =
            {
                .mcts =
                    {
                        .num_searches_per_turn = 640,
                        .num_parallel_searches = 1, // TODO 8?
                        .c_param = 1.7,
                        .dirichlet_alpha = 0.3,
                        .dirichlet_epsilon = 0.25,
                    },
                .num_parallel_games = 32,
                .num_moves_after_which_to_play_greedy = 25,
                .max_moves = 250,
                .result_score_weight = 0.15,
                .resignation_threshold = -1.0,
            },
        .writer =
            {
                .filePrefix = std::string("memory"),
                .batchSize = 5000,
            },
        .inference =
            {
                .maxBatchSize = 128,
            },
    };

    TensorBoardLogger logger(std::string("logs/run_") + std::to_string(runId) +
                             std::string("/tfevents"));

    auto [currentModelPath, currentIteration] =
        get_latest_iteration_save_path(TRAINING_ARGS.save_path);

    // TODO
    // std::vector<InferenceClient> clients(numGPUs * 2);
    // for (int i : range(numGPUs * 2)) { // start 2 InferenceClients per GPU
    //     clients[i].init(i % numGPUs, currentModelPath, TRAINING_ARGS.inference.maxBatchSize,
    //                     &logger);
    // }
    std::vector<InferenceClient> clients(numGPUs);
    for (int i : range(numGPUs)) {
        clients[i].init(i % numGPUs, currentModelPath, TRAINING_ARGS.inference.maxBatchSize,
                        &logger);
    }

    SelfPlayWriter writer(TRAINING_ARGS, logger);
    writer.updateIteration(currentIteration);

    log("Number of processors:", numProcessors, "Number of GPUs:", numGPUs);
    log("Starting on run", runId, "with model path:", currentModelPath,
        "Iteration:", currentIteration);

    std::vector<std::thread> threads;
    for (int i : range(numProcessors)) {
        threads.emplace_back(std::thread([&] {
            SelfPlay selfPlay(&clients[i % clients.size()], &writer, TRAINING_ARGS.self_play,
                              &logger);
            log("Worker process", i + 1, "of", numProcessors, "started");

            while (true) {
                selfPlay.selfPlay();
            }
        }));
    }

    while (true) {
        const auto [latestModelPath, latestIteration] =
            get_latest_iteration_save_path(TRAINING_ARGS.save_path);

        if (latestModelPath == currentModelPath) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            continue;
        }
        log("New model found:", latestModelPath, "Iteration:", latestIteration);
        log("Updating model for all clients");
        for (auto &client : clients) {
            client.updateModel(latestModelPath, latestIteration);
        }

        writer.updateIteration(latestIteration);

        reset_times(&logger, currentIteration);

        currentModelPath = latestModelPath;
        currentIteration = latestIteration;
        log("Model updated for all clients");
    }

    log("Main thread finished");
}

std::vector<PyInferenceResult> boardInferenceMain(const std::string &modelPath,
                                                  const std::vector<std::string> &fens) {
    log("Model path:", modelPath);
    log("Starting board inference");

    const MCTSParams EVALUATION_MCTS_PARAMS = {
        .num_searches_per_turn =
            64, // Reduced number of searches to speed up the evaluation process
        .num_parallel_searches = 8,
        .c_param = 1.7,
        .dirichlet_alpha = 0.3,
        .dirichlet_epsilon = 0.0, // No Dirichlet noise for board inference (greedy)
    };

    const int numBoards = fens.size();
    const int maxBatchSize = numBoards * EVALUATION_MCTS_PARAMS.num_parallel_searches;

    InferenceClient inferenceClient;
    inferenceClient.init(0, modelPath, maxBatchSize, nullptr);
    const MCTS mcts(&inferenceClient, EVALUATION_MCTS_PARAMS, nullptr);

    std::vector<Board> boards;
    boards.reserve(numBoards);
    for (const auto &fen : fens) {
        const Board board = Board::fromFEN(fen);

        if (!board.isValid()) {
            log("Invalid Board fen:", fen);
            throw std::runtime_error("Invalid Board fen: " + fen);
        }
        boards.push_back(board);
    }

    std::vector<PyInferenceResult> pyResults;
    pyResults.reserve(numBoards);

    for (const auto &[score, visits] : mcts.search(boards)) {
        pyResults.emplace_back(score, visits);
    }

    log("Board inference completed");
    return pyResults;
}

Move evalBoardIterate(const std::string &modelPath, const std::string &fen, bool networkOnly,
                      float maxTime) {
    log("Model path:", modelPath);
    log("Starting board inference");
    log("FEN:", fen);
    log("Network only:", networkOnly);
    log("Max time:", maxTime);

    const MCTSParams EVALUATION_MCTS_PARAMS = {
        .num_searches_per_turn =
            2 << 30, // Set to a very high number to ensure we search until maxTime is reached
        .num_parallel_searches = 8,
        .c_param = 1.7,
        .dirichlet_alpha = 0.3,
        .dirichlet_epsilon = 0.0, // No Dirichlet noise for board inference (greedy)
    };

    const int maxBatchSize = 1; // * EVALUATION_MCTS_PARAMS.num_parallel_searches;
    InferenceClient inferenceClient;
    inferenceClient.init(0, modelPath, maxBatchSize, nullptr);
    const MCTS mcts(&inferenceClient, EVALUATION_MCTS_PARAMS, nullptr);

    Board board = Board::fromFEN(fen);
    if (!board.isValid()) {
        log("Invalid Board fen:", fen);
        throw std::runtime_error("Invalid Board fen: " + fen);
    }

    std::vector<Board> boards;
    boards.push_back(board);

    if (networkOnly) {
        // Run the network inference only
        const auto [moves, value] = inferenceClient.inference_batch(boards)[0];
        log("Network inference completed");
        int bestMoveIndex = -1;
        float bestMoveScore = -1.0f;
        for (const auto &[move, score] : moves) {
            if (score > bestMoveScore) {
                bestMoveScore = score;
                bestMoveIndex = move;
            }
        }
        assert(bestMoveIndex != -1);
        log("Best move:", decodeMove(bestMoveIndex), "Best move score:", bestMoveScore);
        log("Result:", value);
        return decodeMove(bestMoveIndex);
    }

    // Run MCTS search until maxTime is reached
    auto start = std::chrono::high_resolution_clock::now();
    auto isTimeUp = [&]() {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start);
        return duration.count() >= maxTime;
    };

    std::vector<MCTSNode> roots;
    roots.push_back(MCTSNode::root(board));

    while (!isTimeUp() &&
           roots[0].number_of_visits < EVALUATION_MCTS_PARAMS.num_searches_per_turn) {
        // Run MCTS search
        mcts.parallel_iterate(roots);
    }

    // Get the best move from the MCTS results
    int bestMoveIndex = -1;
    int bestMoveVisits = -1;
    for (const MCTSNode &child : roots[0].children) {
        if (child.number_of_visits > bestMoveVisits) {
            bestMoveVisits = child.number_of_visits;
            bestMoveIndex = child.encoded_move_to_get_here;
        }
    }
    assert(bestMoveIndex != -1);
    log("Best move:", decodeMove(bestMoveIndex), "Best move visits:", bestMoveVisits);
    log("Result:", roots[0].result_score);
    log("Board inference completed");
    log("Total number of searches:", roots[0].number_of_visits);
    log("Best child visits:", bestMoveVisits);

    return decodeMove(bestMoveIndex);
}
