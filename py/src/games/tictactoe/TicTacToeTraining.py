import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from src.Network import Network
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.games.tictactoe.TicTacToeBoard import TicTacToeBoard
from src.games.tictactoe.TicTacToeGame import TicTacToeGame
from src.settings import TRAINING_ARGS, TensorboardWriter
from src.train.Trainer import Trainer

# --------------------------
# 1. Dataset Preparation
# --------------------------

NUM_EPOCHS = 30
BATCH_SIZE = 256
DATASET_PERCENTAGE = 0.8


# --------------------------
# 2. Model Definition
# --------------------------


class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        # Shared layers
        self.backbone = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Linear(64, 9)

        # Value head with tanh activation
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.backbone(x)

        # Policy output
        policy = self.policy_head(x)  # Raw logits

        # Value output with tanh activation
        value = torch.tanh(self.value_head(x))  # Constrained to [-1, 1]

        return policy, value


# --------------------------
# 3. Training Loop
# --------------------------
def get_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4, amsgrad=True)


def train_model(model, dataloader, device, num_epochs, iteration):
    optimizer = get_optimizer(model)
    trainer = Trainer(model, optimizer, TRAINING_ARGS.training)

    print('Training with lr:', TRAINING_ARGS.training.learning_rate(iteration))

    with TensorboardWriter():
        for epoch in range(num_epochs):
            trainer.train(dataloader, iteration)
            print(f'Epoch {epoch+1}/{num_epochs} done')


def train_model2(model, dataloader, device, num_epochs=20):
    # Define optimizer
    optimizer = get_optimizer(model)

    for epoch in range(num_epochs):
        model.train()
        total_policy_loss = torch.tensor(0.0)
        total_value_loss = torch.tensor(0.0)
        total_loss = torch.tensor(0.0)
        num_batches = 0

        for batch in dataloader:
            board, moves, outcome = batch
            board = board.to(device)
            moves = moves.to(device)
            outcome = outcome.to(device).unsqueeze(1)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            policy_output, value_output = model(board)

            # Compute losses
            policy_loss = F.binary_cross_entropy_with_logits(policy_output, moves)
            value_loss = F.mse_loss(value_output, outcome)
            loss = policy_loss + value_loss  # You can weight them if desired

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate losses
            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_loss += loss
            num_batches += 1

        avg_policy_loss = total_policy_loss.item() / num_batches
        avg_value_loss = total_value_loss.item() / num_batches
        avg_total_loss = total_loss.item() / num_batches

        print(
            f'Epoch [{epoch+1}/{num_epochs}], '
            f'Policy Loss: {avg_policy_loss:.4f}, '
            f'Value Loss: {avg_value_loss:.4f}, '
            f'Total Loss: {avg_total_loss:.4f}'
        )

    # Save the model after training
    torch.save(model.state_dict(), 'tictactoe_model.pth')
    print('Model saved to tictactoe_model.pth')


# --------------------------
# 4. Evaluation
# --------------------------


def evaluate_model(model, dataloader, device):
    dataset = SelfPlayDataset.load('reference/memory_0_tictactoe_database.hdf5')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    model.eval()
    total_policy_correct = 0
    total_policy_total = 0
    total_value_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            board, moves, outcome = batch
            board = board.to(device)
            moves = moves.to(device)
            outcome = outcome.to(device).unsqueeze(1)

            policy_output, value_output = model(board)

            # Policy evaluation
            policy_pred = torch.softmax(policy_output, dim=1)
            # get number (k) of moves in each batch
            # check the top k moves in the policy_pred
            # sum up how many of the top k probabilities are in the moves

            for i in range(len(moves)):
                k = moves[i].sum().to(dtype=torch.int).item()
                top_k = policy_pred[i].topk(k).indices
                total_policy_correct += (top_k == moves[i].nonzero().squeeze()).sum().item()
                total_policy_total += k

            # Value evaluation
            value_loss = F.mse_loss(value_output, outcome)
            total_value_loss += value_loss.item()

    policy_accuracy = total_policy_correct / total_policy_total
    avg_value_loss = total_value_loss / len(dataset)

    print(f'Policy Accuracy: {policy_accuracy*100:.2f}%, Value MSE Loss: {avg_value_loss:.4f}')


# --------------------------
# 5. Inference
# --------------------------


def predict(model, board_state, device):
    """
    Predicts the policy and value for a given board state.

    Args:
        model (nn.Module): Trained model.
        board_state (tuple): Tuple of 9 integers representing the board.
        device (torch.device): Device to run the model on.

    Returns:
        policy_probs (list): List of probabilities for each move.
        value (float): Predicted outcome.
    """
    model.eval()
    with torch.no_grad():
        board_tensor = torch.tensor(board_state, dtype=torch.float32).unsqueeze(0).to(device)
        policy_output, value_output = model(board_tensor)

        # Convert policy logits to probabilities using sigmoid
        policy_probs = torch.softmax(policy_output, dim=1).cpu().detach().numpy()

        # Convert value to float
        value = value_output.mean().item()

    return policy_probs, value


# --------------------------
# 6. Main Execution
# --------------------------

if __name__ == '__main__':
    # python -m AIZeroConnect4Bot.src.games.tictactoe.TicTacToeTraining

    # Instantiate the dataset
    dataset = SelfPlayDataset.load('reference/memory_0_tictactoe_database.hdf5')
    train, test = torch.utils.data.random_split(
        dataset, [int(DATASET_PERCENTAGE * len(dataset)), len(dataset) - int(DATASET_PERCENTAGE * len(dataset))]
    )

    # Create a DataLoader
    train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Instantiate the model
    model = Network(8, 32, device=device)

    # Train the model
    print('Starting training...')
    for iter in range(NUM_EPOCHS):
        if False:
            train_model2(model, train_dataloader, device, num_epochs=1)
        else:
            train_model(model, train_dataloader, device, num_epochs=1, iteration=iter)

        # Evaluate the model
        evaluate_model(model, test_dataloader, device)

    # --------------------------
    # 7. Inference Example
    # --------------------------

    board = TicTacToeBoard()
    game = TicTacToeGame()  # Assuming this class manages the game state

    while not board.is_game_over():
        if board.current_player == -1:
            # Human player
            print('Current Board:')
            print(board.board.reshape(3, 3))
            try:
                move = int(input('Enter your move (0-8): '))
                if move not in board.get_valid_moves():
                    print('Invalid move. Try again.')
                    continue
                board.make_move(move)
            except ValueError:
                print('Invalid input. Please enter an integer between 0 and 8.')
            continue

        # Get the canonical board state if necessary
        # Assuming get_canonical_board returns a normalized board state
        canonical_board = game.get_canonical_board(board)

        # Get the model's predictions
        policy_probs, value = predict(model, canonical_board, device)

        # Mask invalid moves
        valid_moves_mask = game.encode_moves(board.get_valid_moves())
        policy_probs = policy_probs * valid_moves_mask

        # If no valid moves left, it's a draw
        if policy_probs.sum() == 0:
            print("No valid moves left. It's a draw!")
            break

        # Normalize the policy probabilities
        policy_probs /= policy_probs.sum()

        # Select the move with the highest probability
        move = policy_probs.argmax()

        # print rounded policy and value
        print('Policy:', np.round(policy_probs, 2))
        print('Value:', round(value, 2))

        # Make the move
        board.make_move(move)
        print(f'Move made: {move}')
