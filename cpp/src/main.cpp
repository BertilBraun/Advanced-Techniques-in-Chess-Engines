#include "common.hpp"

#include "MCTS/MCTS.hpp"
#include "SelfPlay/SelfPlay.hpp"
#include "SelfPlay/SelfPlayWriter.hpp"

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
            .writer =
                {
                    .filePrefix = 'batch',
                    .batchSize = 5000,
                },
            .num_parallel_games = 32,
            .num_moves_after_which_to_play_greedy = 25,
            .max_moves = 250,
            .result_score_weight = 0.15,
            .num_games_after_which_to_write = 5,
            .resignation_threshold = -1.0,
        },
};

int main() {
    auto numProcessors = std::thread::hardware_concurrency();
    auto numGPUs = torch::cuda::device_count();

    std::vector<InferenceClient> clients;
    for (int i = 0; i < numGPUs * 2; i++) { // start 2 InferenceClients per GPU
        clients.emplace_back(i % numGPUs);
    }

    SelfPlayWriter writer(TRAINING_ARGS.self_play);

    std::vector<std::thread> threads;

    for (size_t i = 0; i < numProcessors; ++i) {
        threads.emplace_back(std::thread([&] {
            SelfPlay selfPlay(&clients[i % clients.size()], &writer, TRAINING_ARGS.self_play);
            log("Worker process", i + 1, "of", numProcessors, "started");

            while (true) {
                timeit([&] { selfPlay.selfPlay(); }, "selfPlay");
            }
        }));
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