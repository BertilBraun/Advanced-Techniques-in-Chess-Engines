from torch.optim import Adam

from AIZeroConnect4Bot.settings import TRAINING_ARGS
from AIZeroConnect4Bot.src.AlphaZero import AlphaZero
from AIZeroConnect4Bot.src.Network import Network


if __name__ == '__main__':
    model = Network()
    optimizer = Adam(model.parameters(), lr=0.02, weight_decay=1e-4)

    AlphaZero(model, optimizer, TRAINING_ARGS).learn()
