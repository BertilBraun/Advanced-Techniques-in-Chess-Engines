# Implementation Details

We only provide a brief overview of the key components and classes in the AlphaZero-Clone project. For a more detailed explanation, please refer to the source code and documentation in the repository. Since the project is under active development, the implementation details may change over time.

In general, the main implementation is currently in Python using PyTorch. Different [Architectures](optimizations/architecture.md) and [Optimizations](../README.md#optimizations) are tried out and documented in the repository. The project is planned to be translated to C++ for better scalability and performance.

## Neural Network Model

The core of the AlphaZero-Clone project is the neural network model responsible for predicting move probabilities (policy) and evaluating board states (value). This model is based on a residual neural network architecture with configurable depth and hidden layer sizes. The network processes the game state representation and outputs policy and value predictions for the current board state.

```python
class Network(nn.Module):
    def __init__(self, num_res_blocks: int, hidden_size: int, device: torch.device) -> None:
        encoding_channels, row_count, column_count = CurrentGame.representation_shape
        action_size = CurrentGame.action_size

        self.startBlock = nn.Sequential(
            nn.Conv2d(encoding_channels, hidden_size, kernel_size=3, padding='same'),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
        )

        self.backBone = nn.ModuleList([ResBlock(hidden_size) for _ in range(num_res_blocks)])

        self.policyHead = nn.Sequential(
            nn.Conv2d(hidden_size, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (row_count - 2) * (column_count - 2), action_size),
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(hidden_size, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (row_count - 2) * (column_count - 2), 1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        return self.policyHead(x), self.valueHead(x)
        
class ResBlock(nn.Module):
    def __init__(self, num_hidden: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(num_hidden) 

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual) 
```

## Abstract Game and Board Classes

To ensure flexibility and support for multiple games, abstract base classes for `Game` and `Board` are defined. These classes enforce a consistent interface across different game implementations. To implement a new game, one must subclass these abstract classes and provide the necessary methods. Other than that, only hyperparameters and game-specific logic need to be defined, the rest of the pipeline remains the same.

```python
class Game(ABC, Generic[_Move]):
    @property
    @abstractmethod
    def null_move(self) -> _Move:
        pass

    @property
    @abstractmethod
    def action_size(self) -> int:
        """The number of possible moves in the game."""
        pass

    @property
    @abstractmethod
    def representation_shape(self) -> tuple[int, int, int]:
        """(num_channels, height, width)"""
        pass

    @property
    @abstractmethod
    def average_num_moves_per_game(self) -> int:
        pass

    @abstractmethod
    def get_canonical_board(self, board: Board[_Move]) -> np.ndarray:
        """Returns a canonical representation of the board from the perspective of the current player.
        No matter the current player, the board should always be from the perspective as if the player to move is 1.
        The board should be a numpy array with shape (num_channels, height, width) as returned by the `representation_shape` property."""
        pass

    @abstractmethod
    def encode_move(self, move: _Move) -> int:
        """Encodes a move into the index of the action in the policy vector."""
        pass

    @abstractmethod
    def decode_move(self, move: int) -> _Move:
        """Decodes an action index into a move."""
        pass

    @abstractmethod
    def hash_boards(self, boards: torch.Tensor) -> list[int]:
        """Hashes a batch of encoded canonical boards.
        The input tensor has shape (batch_size, num_channels, height, width) also (batch_size, *representation_shape)."""
        pass

    @abstractmethod
    def symmetric_variations(
        self, board: np.ndarray, action_probabilities: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Returns a list of symmetric variations of the board and the corresponding action probabilities.
        The board is a numpy array with shape (num_channels, height, width) as returned by the `representation_shape` property.
        The action_probabilities is a numpy array with shape (action_size)."""
        pass

    @abstractmethod
    def get_initial_board(self) -> Board[_Move]:
        pass

class Board(ABC, Generic[_Move]):
    def __init__(self) -> None:
        self.current_player: Player = 1

    @abstractmethod
    def make_move(self, move: _Move) -> None:
        pass

    @abstractmethod
    def is_game_over(self) -> bool:
        pass

    @abstractmethod
    def check_winner(self) -> Optional[Player]:
        pass

    @abstractmethod
    def get_valid_moves(self) -> list[_Move]:
        pass

    @abstractmethod
    def copy(self) -> Board[_Move]:
        pass

    @abstractmethod
    def quick_hash(self) -> int:
        pass
```

## Key Components Explained

- **Network Class:** Defines the neural network architecture with a configurable number of residual blocks. It processes the game state and outputs both policy and value predictions.
- **ResBlock Class:** Implements a residual block with two convolutional layers and batch normalization, facilitating deeper networks by mitigating vanishing gradient issues.
- **Game and Board Classes:** Abstract base classes that enforce a standard interface for different games, ensuring that the core components like move encoding, board representation, and game mechanics are consistently implemented.
