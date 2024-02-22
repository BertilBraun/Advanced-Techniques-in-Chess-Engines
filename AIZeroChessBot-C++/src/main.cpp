#include "common.hpp"

#include "AlphaZeroSelfPlayer.hpp"
#include "AlphaZeroTrainer.hpp"
#include "Network.hpp"
#include "StockfishDataGenerator.hpp"
#include "TrainingArgs.hpp"

std::string SAVE_PATH = "models";

int main(int argc, char *argv[]) {

    // zeroth argument should be the name of the program
    // first argument should be either "root" or "worker"
    // second argument should be the rank of the process
    // third argument should be the number of processes

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " [root|worker|generator] <rank> <numProcesses>"
                  << std::endl;
        return 1;
    }

    bool isRoot = std::string(argv[1]) == "root";
    bool isWorker = std::string(argv[1]) == "worker";
    bool isGenerator = std::string(argv[1]) == "generator";

    size_t rank = std::stoul(argv[2]);
    size_t numProcesses = std::stoul(argv[3]);

    if (!isRoot && !isWorker && !isGenerator) {
        std::cerr << "Invalid argument: " << argv[1] << std::endl;
        return 1;
    }

    TrainingArgs args{
        200,      // numIterations
        32,       // numParallelGames
        500,      // numIterationsPerTurn
        40,       // numEpochs
        64,       // batchSize
        1.0f,     // temperature
        0.25f,    // dirichletEpsilon
        0.03f,    // dirichletAlpha
        2.0f,     // cParam
        "models", // savePath
        75        // retentionRate (in percent)
    };

    if (isGenerator) {
        std::cerr << "Data generator process started" << std::endl;

        // Run the python script in ../src/PreprocessGenerationData.py
        // to generate the data in the correct format

        // Subprocess preprocessData("python3 ../src/PreprocessGenerationData.py", "r+");
        // preprocessData << "data/Lichess Elite Database";
        // preprocessData << "data/lichess_elites.txt";
        // preprocessData << "data/lichess_db_eval.json";

        StockfishDataGenerator stockfishDataGenerator(args);
        stockfishDataGenerator.generateDataFromLichessEval("data/lichess_db_eval.json", false, rank,
                                                           numProcesses);
        stockfishDataGenerator.generateDataFromEliteGames("data/lichess_elites.txt");
        stockfishDataGenerator.generateDataThroughStockfishSelfPlay("models/stockfish_8_x64");
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

    Network model;
    auto optimizer =
        torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(0.02).weight_decay(1e-4));

    if (isRoot) {
        std::cerr << "Trainer process started" << std::endl;
        AlphaZeroTrainer alphaZeroTrainer(model, optimizer, args);
        alphaZeroTrainer.run();
    } else {
        std::cerr << "Worker process " << rank << " of " << numProcesses << " started" << std::endl;
        AlphaZeroSelfPlayer alphaZeroSelfPlayer(rank, model, args);
        alphaZeroSelfPlayer.run();
    }

    return 0;
}
