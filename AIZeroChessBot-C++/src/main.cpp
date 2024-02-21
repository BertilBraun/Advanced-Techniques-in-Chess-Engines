#include "common.hpp"

#include "AlphaZeroSelfPlayer.hpp"
#include "AlphaZeroTrainer.hpp"
#include "Network.hpp"
#include "TrainingArgs.hpp"

int main(int argc, char *argv[]) {

    // zeroth argument should be the name of the program
    // first argument should be either "root" or "worker"
    // second argument should be the rank of the process
    // third argument should be the number of processes

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " [root|worker] <rank> <numProcesses>" << std::endl;
        return 1;
    }

    bool isRoot = std::string(argv[1]) == "root";
    size_t rank = std::stoul(argv[2]);
    size_t numProcesses = std::stoul(argv[3]);

    Network model;
    auto optimizer =
        torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(0.02).weight_decay(1e-4));

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

    TrainingArgs args{
        200,      // numIterations
        32,       // numParallelGames
        200,      // numIterationsPerTurn
        30,       // numEpochs
        64,       // batchSize
        1.0f,     // temperature
        0.25f,    // dirichletEpsilon
        0.03f,    // dirichletAlpha
        2.0f,     // cParam
        "models", // savePath
        75        // retentionRate (in percent)
    };

    if (isRoot) {
        std::cout << "Trainer process started" << std::endl;
        AlphaZeroTrainer alphaZeroTrainer(model, optimizer, args);
        alphaZeroTrainer.run();
    } else {
        std::cout << "Worker process " << rank << " of " << numProcesses << " started" << std::endl;
        AlphaZeroSelfPlayer alphaZeroSelfPlayer(rank, model, args);
        alphaZeroSelfPlayer.run();
    }

    return 0;
}
