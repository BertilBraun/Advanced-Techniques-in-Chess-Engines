import torch
from torch.utils.data import Dataset, DataLoader
import ast
import torch.nn as nn

from src.Network import Network
from src.games.tictactoe.TicTacToeBoard import TicTacToeBoard
from src.games.tictactoe.TicTacToeGame import TicTacToeGame
from src.mcts.MCTS import SelfPlayMemory
from src.settings import TRAINING_ARGS, VALUE_OUTPUT_HEADS
from src.train.Trainer import Trainer

# --------------------------
# 1. Dataset Preparation
# --------------------------

NUM_VALUE_OUTPUTS = VALUE_OUTPUT_HEADS
NUM_EPOCHS = 50
BATCH_SIZE = 16
DATASET_PERCENTAGE = 0.8


class TicTacToeDataset(Dataset):
    def __init__(self, file_path: str):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                board_str, moves_str, outcome_str = line.strip().split(';')

                # Convert board string to list of integers
                board = ast.literal_eval(board_str)
                board_tensor = torch.tensor(board, dtype=torch.float32).reshape(1, 3, 3)

                # Convert moves string to list of integers and then to multi-hot encoding
                moves = ast.literal_eval(moves_str)
                moves_tensor = torch.zeros(9, dtype=torch.float32)
                for move in moves:
                    moves_tensor[move] = 1.0  # Multi-hot encoding

                # Convert outcome to float (-1, 0, 1)
                outcome = torch.tensor([float(outcome_str)] * NUM_VALUE_OUTPUTS, dtype=torch.float32)

                self.data.append((board_tensor, moves_tensor, outcome))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


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
        self.value_head = nn.Linear(64, NUM_VALUE_OUTPUTS)

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


def train_model(model, dataloader, device, num_epochs, iteration):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model, optimizer, TRAINING_ARGS)

    print('Training with lr:', TRAINING_ARGS.learning_rate(iteration))

    self_play_memories: list[SelfPlayMemory] = []
    for batch in dataloader:
        board, moves, outcome = batch
        for b, m, o in zip(board, moves, outcome):
            self_play_memories.append(SelfPlayMemory(b, m, o))
            # print(f'Added memory: {b}, {m}, {o}')

    for epoch in range(num_epochs):
        trainer.train(self_play_memories, iteration)
        print(f'Epoch {epoch+1}/{num_epochs} done')


def train_model2(model, dataloader, device, num_epochs=20):
    # Define loss functions
    policy_loss_fn = nn.BCEWithLogitsLoss()
    value_loss_fn = nn.MSELoss()

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            board, moves, outcome = batch
            board = board.to(device)
            moves = moves.to(device)
            outcome = outcome.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            policy_output, value_output = model(board)

            # Compute losses
            policy_loss = policy_loss_fn(policy_output, moves)
            value_loss = value_loss_fn(value_output, outcome)
            loss = policy_loss + value_loss  # You can weight them if desired

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate losses
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()
            num_batches += 1

        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_total_loss = total_loss / num_batches

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
    model.eval()
    total_policy_correct = 0
    total_policy_total = 0
    total_value_loss = 0.0
    value_loss_fn = nn.MSELoss()

    with torch.no_grad():
        for batch in dataloader:
            board, moves, outcome = batch
            board = board.to(device)
            moves = moves.to(device)
            outcome = outcome.to(device)

            policy_output, value_output = model(board)

            # Policy evaluation
            policy_pred = torch.sigmoid(policy_output)
            policy_pred_binary = (policy_pred > 0.5).float()
            total_policy_correct += (policy_pred_binary == moves).sum().item()
            total_policy_total += moves.numel()

            # Value evaluation
            value_loss = value_loss_fn(value_output, outcome)
            total_value_loss += value_loss.item()

    policy_accuracy = total_policy_correct / total_policy_total
    avg_value_loss = total_value_loss / len(dataloader)

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
        policy_probs = torch.sigmoid(policy_output).squeeze().cpu().detach().numpy()

        # Convert value to float
        value = value_output.mean().item()

    return policy_probs, value


# --------------------------
# 6. Main Execution
# --------------------------

if __name__ == '__main__':
    # python -m AIZeroConnect4Bot.src.games.tictactoe.TicTacToeTraining

    # Path to your dataset file
    dataset_path = 'tictactoe_database.txt'

    # Instantiate the dataset
    dataset = TicTacToeDataset(dataset_path)
    train, test = torch.utils.data.random_split(
        dataset, [int(DATASET_PERCENTAGE * len(dataset)), len(dataset) - int(DATASET_PERCENTAGE * len(dataset))]
    )

    # Create a DataLoader
    train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

    # Instantiate the model
    model = Network(4, 64)

    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Train the model
    print('Starting training...')
    for iter in range(NUM_EPOCHS):
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
        board_tensor = canonical_board  # Already in tuple format
        print('Canonical Board:')
        print(canonical_board.reshape(3, 3))

        # Get the model's predictions
        policy_probs, value = predict(model, board_tensor, device)

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
        print('Policy:', [round(p, 2) for p in policy_probs])
        print('Value:', round(value, 2))

        # Make the move
        board.make_move(move)
        print(f'Move made: {move}')
