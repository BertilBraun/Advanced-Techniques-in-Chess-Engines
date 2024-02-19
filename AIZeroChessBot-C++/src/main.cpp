#include "common.hpp"

#include "AlphaZero.hpp"
#include "Network.hpp"
#include "TrainingArgs.hpp"

int main() {

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

    args = TrainingArgs(
        num_iterations=200,
        num_self_play_iterations=512,
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
        200,     // numIterations
        4,       // numSelfPlayIterations
        2,       // numParallelGames            // TODO test with 2
        5,       // numIterationsPerTurn
        2,       // numEpochs
        1,       // numSeparateNodesOnCluster   // TODO test with 2
        64,      // batchSize
        1.0f,    // temperature
        0.25f,   // dirichletEpsilon
        0.03f,   // dirichletAlpha
        2.0f,    // cParam
        "models" // savePath
    };

    AlphaZero alphaZero(model, optimizer, args);
    alphaZero.learn();

    return 0;
}
