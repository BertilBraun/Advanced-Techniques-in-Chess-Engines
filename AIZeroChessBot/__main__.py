from torch.optim import Adam

from AIZeroChessBot.src.AlphaZero import AlphaZero
from AIZeroChessBot.src.Network import Network
from AIZeroChessBot.src.TrainingArgs import TrainingArgs


if __name__ == '__main__':
    model = Network()
    optimizer = Adam(model.parameters(), lr=0.02, weight_decay=1e-4)

    ALPHA_ZERO_TRAINING_ARGS = TrainingArgs(
        num_iterations=200,
        num_self_play_iterations=500_000,
        num_parallel_games=128,  # unknown
        num_iterations_per_turn=1600,
        num_epochs=20,  # unknown
        batch_size=2048,
        temperature=1.0,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.03,
        c_param=2.0,  # unknown
    )

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
        num_self_play_iterations=500,
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

    AlphaZero(model, optimizer, args).learn()
