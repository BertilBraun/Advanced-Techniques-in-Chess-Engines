#include "common.hpp"

#include "MCTS/MCTS.hpp"
#include "SelfPlay/SelfPlay.hpp"
#include "SelfPlay/SelfPlayWriter.hpp"

int selfPlayMain(const std::vector<std::string> &args);
int boardInferenceMain(const std::vector<std::string> &args);

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <command> [args...]" << std::endl;
        return 1;
    }

    std::string command = argv[1];
    std::vector<std::string> args(argv + 1, argv + argc);

    if (command == "selfplay") {
        return selfPlayMain(args);
    } else if (command == "board_inference") {
        return boardInferenceMain(args);
    } else {
        std::cerr << "Unknown command: " << command << std::endl;
        return 1;
    }
}

int selfPlayMain(const std::vector<std::string> &args) {
    if (args.size() != 4) {
        std::cerr << "Usage: selfplay <runId> <savePath> <numProcessors> <numGPUs>" << std::endl;
        return 1;
    }

    const int runId = std::stoi(args[0]);
    assert(runId >= 0);
    const std::string savePath = args[1];
    const int numProcessors = std::stoi(args[2]);
    assert(numProcessors >= 1);
    const int numGPUs = std::stoi(args[3]);
    assert(0 < numGPUs && numGPUs < torch::cuda::device_count());

    const TrainingArgs TRAINING_ARGS = {
        .save_path = savePath,
        .self_play =
            {
                .mcts =
                    {
                        .num_searches_per_turn = 640,
                        .num_parallel_searches = 8,
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
                .filePrefix = std::string("batch"),
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

    std::vector<InferenceClient> clients;
    for (int i : range(numGPUs * 2)) { // start 2 InferenceClients per GPU
        clients.emplace_back(i % numGPUs, currentModelPath, TRAINING_ARGS.inference.maxBatchSize,
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

        reset_times(logger, currentIteration);

        currentModelPath = latestModelPath;
        currentIteration = latestIteration;
        log("Model updated for all clients");
    }

    log("Main thread finished");
    return 0;
}

int boardInferenceMain(const std::vector<std::string> &args) {
    if (args.size() != 2) {
        std::cerr << "Usage: board_inference <modelPath> <board_fens ...>" << std::endl;
        return 1;
    }

    const std::string modelPath = args[0];

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

    const int numBoards = args.size() - 1;
    const int maxBatchSize = numBoards * EVALUATION_MCTS_PARAMS.num_parallel_searches;

    InferenceClient inferenceClient(0, modelPath, maxBatchSize, nullptr);
    const MCTS mcts(&inferenceClient, EVALUATION_MCTS_PARAMS, nullptr);

    std::vector<Board> boards;
    boards.reserve(numBoards);
    for (int i : range(1, (int) args.size())) {
        const Board board = Board::fromFEN(args[i]);

        if (!board.isValid()) {
            log("Invalid Board fen:", args[i]);
            return 1;
        }
        boards.push_back(board);
    }

    for (const auto &[score, visits] : mcts.search(boards)) {
        std::cout << score << ":";
        for (const auto &[move, count] : visits.visits) {
            std::cout << "(" << move << "," << count << ")";
        }
        std::cout << std::endl;
    }

    return 0;
}
