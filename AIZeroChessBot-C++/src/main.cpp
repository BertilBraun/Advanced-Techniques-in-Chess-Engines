#include "common.hpp"

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
        40,        // numEpochs
        64,        // batchSize
        1.0f,      // temperature
        0.25f,     // dirichletEpsilon
        0.03f,     // dirichletAlpha
        2.0f,      // cParam
        SAVE_PATH, // savePath
        75,        // retentionRate (in percent)
        1000000,   // numTrainers
    };

    if (isGenerate) {
        log("Data generator process started");

        // Run the python script in ../src/PreprocessGenerationData.py
        // to generate the data in the correct format

        // Subprocess preprocessData("python3 ../src/PreprocessGenerationData.py", "r+");
        // preprocessData << "data/Lichess Elite Database";
        // preprocessData << "data/lichess_elites.txt";
        // preprocessData << "data/lichess_db_eval.json";

        std::vector<std::thread> threads;

        for (size_t i = 0; i < numProcessors; ++i) {
            threads.emplace_back(std::thread([i, numProcessors, args] {
                StockfishDataGenerator stockfishDataGenerator(args.batchSize);
                stockfishDataGenerator.generateDataFromLichessEval("data/lichess_db_eval.json",
                                                                   false, i, numProcessors);
                // stockfishDataGenerator.generateDataFromEliteGames("data/lichess_elites.txt");
                // stockfishDataGenerator.generateDataThroughStockfishSelfPlay(
                //     "models/stockfish_8_x64");
            }));
        }

        for (auto &thread : threads) {
            thread.join();
        }

        return 0;
    }

    /*
    args = TrainingArgs(
        num_iterations=200,
        num_self_play_iterations=32000,
        num_parallel_games=64,
        num_iterations_per_turn=200,
        num_epochs=20,
        num_separate_nodes_on_cluster=2,
        batch_size=64,
        temperature=1.0,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.03,
        c_param=2.0,
    )
    */

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
