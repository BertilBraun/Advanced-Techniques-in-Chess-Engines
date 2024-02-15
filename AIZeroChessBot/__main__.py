import torch

from AIZeroChessBot.AlphaZero import AlphaZero
from AIZeroChessBot.Network import Network
from AIZeroChessBot.TrainingArgs import TrainingArgs


if __name__ == '__main__':
    model = Network()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.2, weight_decay=1e-4)

    args = TrainingArgs(
        num_iterations=500,
        num_self_play_iterations=50,
        num_parallel_games=50,
        num_searches=50,
        num_epochs=10,
        batch_size=64,
        temperature=1.0,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.03,
        c_param=2.0,
    )

    AlphaZero(model, optimizer, args).learn()
