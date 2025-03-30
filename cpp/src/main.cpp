#include "common.hpp"

#include "MCTS/MCTS.hpp"
#include "SelfPlay/SelfPlay.hpp"
#include "SelfPlay/SelfPlayWriter.hpp"
#include <cassert>

static inline TrainingArgs TRAINING_ARGS = {
    .save_path = 'training_data/chess',
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
            .filePrefix = 'batch',
            .batchSize = 5000,
        },
    .inference =
        {
            .maxBatchSize = 128,
        },
};

// Assume this function is implemented elsewhere.
// It returns a pair: (latest model file path, iteration number).
std::pair<std::string, int> get_latest_iteration_save_path(const std::string &savePath) {
    // Models are saved in the savePath folder numbered starting from 1 up to the latest iteration.
    // For example: "{savePath}/model_1.pt", "{savePath}/model_2.pt", etc.
    // The function returns the latest model file path and its iteration number.

    for (int i : range(500, 0, -1)) {
        std::string modelPath = fmt::format("{}/model_{}.pt", savePath, i);
        if (std::filesystem::exists(modelPath)) {
            return {modelPath, i};
        }
    }
    assert(false && "No model found in the save path.");
}

int main() {
    int numProcessors = std::thread::hardware_concurrency() - 10; // leave 10 cores for other tasks
    numProcessors = std::max(numProcessors, 2);
    int numGPUs = torch::cuda::device_count();
    numGPUs = std::max(numGPUs, 1);

    TensorBoardLogger logger("logs/tfevents"); // TODO get the run id from sys args

    auto [currentModelPath, currentIteration] =
        get_latest_iteration_save_path(TRAINING_ARGS.save_path);

    std::vector<InferenceClient> clients;
    for (int i = 0; i < numGPUs * 2; i++) { // start 2 InferenceClients per GPU
        clients.emplace_back(i % numGPUs, currentModelPath, TRAINING_ARGS.inference.maxBatchSize,
                             logger);
    }

    SelfPlayWriter writer(TRAINING_ARGS, logger);
    writer.updateIteration(currentIteration);

    std::vector<std::thread> threads;

    for (size_t i = 0; i < numProcessors; ++i) {
        threads.emplace_back(std::thread([&] {
            SelfPlay selfPlay(&clients[i % clients.size()], &writer, TRAINING_ARGS.self_play,
                              logger);
            log("Worker process", i + 1, "of", numProcessors, "started");

            while (true) {
                timeit([&] { selfPlay.selfPlay(); }, "selfPlay");
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

        currentModelPath = latestModelPath;
        currentIteration = latestIteration;
        log("Model updated for all clients");
    }

    for (auto &thread : threads) {
        thread.join();
    }

    return 0;
}

/*
#include "AlphaZeroSelfPlayer.hpp"
#include "AlphaZeroTrainer.hpp"
#include "Network.hpp"
#include "StockfishDataGenerator.hpp"
#include "TrainingArgs.hpp"

int main(int argc, char *argv[]) {

    if (argc != 3) {
        log("Usage:", argv[0], "[train|generate] <numProcessors>");
        return 1;
    }

    bool isTrain = std::string(argv[1]) == "train";
    bool isGenerate = std::string(argv[1]) == "generate";

    size_t numProcessors = std::stoul(argv[2]);

    if (!isTrain && !isGenerate) {
        log("Invalid argument:", argv[1]);
        return 1;
    }

    if (numProcessors == 0) {
        log("Warning: numProcessors is 0");
    }

    TrainingArgs args{
        200,       // numIterations
        32,        // numParallelGames
        500,       // numIterationsPerTurn
        1,         // numEpochs // TODO 40
        64,        // batchSize // TODO 128
        1.0f,      // temperature
        0.25f,     // dirichletEpsilon
        0.03f,     // dirichletAlpha
        2.0f,      // cParam
        SAVE_PATH, // savePath
        100,       // retentionRate (in percent) // TODO 75
        1000000,   // numTrainers // TODO 2-4
    };

    if (isGenerate) {
        log("Data generator process started");

        std::vector<std::thread> threads;

        for (size_t i = 0; i < numProcessors; ++i) {
            threads.emplace_back(std::thread([i, numProcessors, args] {
                StockfishDataGenerator stockfishDataGenerator(args.batchSize);
                // stockfishDataGenerator.generateDataFromLichessEval("data/lichess_db_eval.json",
                //                                                    false, i, numProcessors);
                stockfishDataGenerator.generateDataFromEliteGames("data/lichess_elites.txt",
                                                                  "stockfish-windows-x86-64.exe");
                // stockfishDataGenerator.generateDataThroughStockfishSelfPlay(
                //     "stockfish-windows-x86-64.exe", numProcessors);
            }));
        }

        for (auto &thread : threads) {
            thread.join();
        }

        return 0;
    }


    // args = TrainingArgs(
    //     num_iterations=200,
    //     num_self_play_iterations=32000,
    //     num_parallel_games=64,
    //     num_iterations_per_turn=200,
    //     num_epochs=20,
    //     num_separate_nodes_on_cluster=2,
    //     batch_size=64,
    //     temperature=1.0,
    //     dirichlet_epsilon=0.25,
    //     dirichlet_alpha=0.03,
    //     c_param=2.0,
    // )


    std::vector<std::thread> threads;

    for (size_t i = 0; i < numProcessors; ++i) {
        threads.emplace_back(std::thread([i, numProcessors, args] {
            Network model;
            AlphaZeroSelfPlayer alphaZeroSelfPlayer(model, args);

            log("Worker process", i + 1, "of", numProcessors, "started");

            alphaZeroSelfPlayer.run();
        }));
    }

    Network model;
    auto optimizer =
        torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(1e-3).weight_decay(2e-5));

    AlphaZeroTrainer alphaZeroTrainer(model, optimizer, args);

    log("Trainer process started");

    alphaZeroTrainer.run();

    for (auto &thread : threads) {
        thread.join();
    }

    return 0;
}
*/